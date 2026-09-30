# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
PettingZoo ``ParallelEnv`` around an ASSUME market simulation.

One environment step corresponds to **one market opening**: every externally
controlled unit submits its bid for the same delivery product, the market
clears, and the resulting rewards come back.

See :mod:`assume.reinforcement_learning.pettingzoo.env_coordinator` for how the
simulation is driven, and the module docstring of
:mod:`assume.reinforcement_learning.pettingzoo` for the design rationale.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

from assume.reinforcement_learning.pettingzoo.env_coordinator import (
    ActionRequest,
    EnvCoordinator,
)
from assume.reinforcement_learning.pettingzoo.external_learning_role import (
    ExternalLearning,
)
from assume.reinforcement_learning.pettingzoo.strategies import (
    EXTERNAL_STRATEGY_ALIASES,
    EXTERNAL_STRATEGY_NAMES,
    ExternalControlMixin,
)
from assume.scenario.loader_csv import load_config_and_create_forecaster, setup_world
from assume.world import World

logger = logging.getLogger(__name__)


class AssumeParallelEnv(ParallelEnv):
    """
    Multi-agent environment in which each externally controlled ASSUME unit is
    one PettingZoo agent.

    Agent ids are ASSUME ``unit_id`` strings. Observations are exactly the
    vectors ASSUME's ``create_observation`` builds; actions are exactly the
    vectors ASSUME's ``calculate_bids`` expects (the actor network output
    domain, ``[-1, 1]`` by default); rewards are exactly what ASSUME's
    ``calculate_reward`` computes.

    Args:
        inputs_path: Folder containing the ASSUME scenario inputs.
        scenario: Scenario (sub-folder) name.
        study_case: Study case inside the scenario's ``config.yaml``.
        database_uri: Optional database for ASSUME's regular outputs. Empty by
            default - writing results every episode slows training down a lot.
        export_csv_path: Optional CSV export path. Empty by default.
        action_low: Lower bound of the action space.
        action_high: Upper bound of the action space.
        max_episode_steps: Truncate the episode after this many market
            openings. ``None`` runs the full simulation horizon.
        strict_observations: Raise instead of warning if the pre-computed
            observation of an agent differs from the one the simulation builds
            at that agent's turn (see :ref:`the note on simultaneity <simultaneity>`).
        log_level: Log level handed to :class:`~assume.world.World`.
        world: Use an already-configured :class:`~assume.world.World` whose
            ``scenario_data`` is loaded, instead of ``inputs_path``/``scenario``/
            ``study_case``.
    """

    metadata = {
        "render_modes": ["human"],
        "name": "assume_parallel_v0",
        "is_parallelizable": False,
    }

    def __init__(
        self,
        inputs_path: str | Path | None = None,
        scenario: str | None = None,
        study_case: str | None = None,
        *,
        database_uri: str = "",
        export_csv_path: str = "",
        action_low: float = -1.0,
        action_high: float = 1.0,
        max_episode_steps: int | None = None,
        strict_observations: bool = False,
        show_progress: bool = False,
        log_level: str = "WARNING",
        render_mode: str | None = None,
        world: World | None = None,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.action_low = float(action_low)
        self.action_high = float(action_high)

        if world is None:
            if inputs_path is None or scenario is None or study_case is None:
                raise ValueError(
                    "Provide either `world` or all of `inputs_path`, `scenario` "
                    "and `study_case`."
                )
            world = World(
                database_uri=database_uri,
                export_csv_path=export_csv_path,
                log_level=log_level,
            )
            world.scenario_data = load_config_and_create_forecaster(
                str(inputs_path), scenario, study_case
            )
        elif not world.scenario_data:
            raise ValueError(
                "`world.scenario_data` is empty - load a scenario first, e.g. with "
                "assume.scenario.loader_csv.load_config_and_create_forecaster()."
            )

        self.world = world
        # one ASSUME simulation per RL episode makes the per-run progress bar
        # more noise than signal
        self.world.show_progress = show_progress
        self.coordinator = EnvCoordinator(name="assume-pettingzoo")
        self.coordinator.strict_observations = strict_observations

        self._prepare_world()

        self._episode = 0
        self._steps = 0
        self._blocked: ActionRequest | None = None
        self._cycle_key: tuple[Any, Any] | None = None
        self._roster: dict[str, tuple[Any, Any]] = {}
        self._last_infos: dict[str, dict] = {}

        # discover agents and their observation/action dimensions
        self.possible_agents: list[str] = []
        self._obs_dims: dict[str, int] = {}
        self._act_dims: dict[str, int] = {}
        self._agent_markets: dict[str, frozenset[str]] = {}
        self._observation_spaces: dict[str, spaces.Box] = {}
        self._action_spaces: dict[str, spaces.Box] = {}
        self._bootstrap()

        self.agents: list[str] = list(self.possible_agents)

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def _prepare_world(self) -> None:
        """
        Point the world at the external strategies and the external learning role.

        Both changes are additive: the scenario files, the market configuration
        and every non-learning strategy stay untouched.
        """
        # 0. World.__init__ assigns the *module-level* `bidding_strategies` dict
        #    by reference, so rebinding entries in place would silently affect
        #    every other World in the process. Give this world its own copy.
        self.world.bidding_strategies = dict(self.world.bidding_strategies)

        # 1. swap every learning strategy for its externally controlled twin, so
        #    existing scenarios (`bidding_EOM: powerplant_energy_learning`) work
        #    without editing any CSV
        for name, external_cls in EXTERNAL_STRATEGY_ALIASES.items():
            if name in self.world.bidding_strategies:
                self.world.bidding_strategies[name] = external_cls
        # 2. also expose the explicit names
        self.world.bidding_strategies.update(EXTERNAL_STRATEGY_NAMES)

        # 3. hand ASSUME a Learning role that forwards rewards to us
        self.world.learning_role_class = functools.partial(
            ExternalLearning, coordinator=self.coordinator
        )

        # 4. force the flags the external control path needs. learning_mode makes
        #    the unchanged strategies fill ASSUME's experience caches (that is how
        #    rewards reach us); evaluation_mode stops ASSUME from writing to its
        #    own replay buffer and running MATD3 updates - training belongs to the
        #    external framework now.
        config = self.world.scenario_data.setdefault("config", {})
        learning_config = config.setdefault("learning_config", {})
        if not learning_config:
            raise ValueError(
                "The selected study case has no `learning_config`. Add one to "
                "config.yaml (at minimum `learning_mode: true`) so ASSUME builds "
                "learning strategies for the units you want to control."
            )
        learning_config["learning_mode"] = True
        learning_config["continue_learning"] = False

    def _setup_episode(self) -> None:
        """Build a fresh ASSUME world for one episode. Runs on the sim thread."""
        asyncio.set_event_loop(self.world.loop)
        setup_world(
            world=self.world,
            evaluation_mode=True,
            episode=self._episode,
            eval_episode=self._episode,
        )

    def _run_episode(self) -> None:
        """Full episode target for the coordinator's worker thread."""
        self._setup_episode()
        self.world.run()

    def _bootstrap(self) -> None:
        """
        Build the world once, without running it, to learn the agent roster and
        the observation/action dimensions PettingZoo needs up front.

        The world built here is never run; the first :meth:`reset` rebuilds it.
        """
        self._setup_episode()

        roster = self._build_roster()
        if not roster:
            raise ValueError(
                "No externally controlled units found. Give at least one unit a "
                "learning bidding strategy (e.g. `powerplant_energy_learning`) in "
                "the scenario inputs."
            )

        self.possible_agents = sorted(roster)
        for agent_id, (_unit, strategy) in roster.items():
            self._obs_dims[agent_id] = int(strategy.obs_dim)
            self._act_dims[agent_id] = int(strategy.act_dim)

        market_sets = set(self._agent_markets.values())
        if len(market_sets) > 1:
            raise NotImplementedError(
                "All externally controlled units must bid on the same set of "
                "markets, otherwise a market opening does not involve every agent "
                "and the PettingZoo agent set would change between steps. Found: "
                f"{ {a: sorted(m) for a, m in self._agent_markets.items()} }"
            )
        if len(next(iter(market_sets))) > 1:
            logger.warning(
                "Externally controlled units bid on more than one market (%s). "
                "Each market opening is a separate environment step, so the "
                "agents will be asked once per market per delivery period.",
                sorted(next(iter(market_sets))),
            )

    def _build_roster(self) -> dict[str, tuple[Any, Any]]:
        """
        Collect ``{unit_id: (unit, strategy)}`` for every externally controlled
        unit of the currently set-up world.
        """
        roster: dict[str, tuple[Any, Any]] = {}
        self._agent_markets = {}
        for unit_id, unit in self.world.units.items():
            markets = {
                market_id
                for market_id, strategy in unit.bidding_strategies.items()
                if isinstance(strategy, ExternalControlMixin)
            }
            if not markets:
                continue
            market_id = sorted(markets)[0]
            roster[unit_id] = (unit, unit.bidding_strategies[market_id])
            self._agent_markets[unit_id] = frozenset(markets)
        return roster

    # ------------------------------------------------------------------
    # PettingZoo spaces
    # ------------------------------------------------------------------
    # Note: cached in plain dicts rather than with functools.lru_cache, which
    # would keep a reference to `self` alive for the process lifetime - and each
    # environment owns a full ASSUME world.
    def observation_space(self, agent: str) -> spaces.Box:
        """Unbounded box with the dimension of ASSUME's observation vector."""
        if agent not in self._observation_spaces:
            self._observation_spaces[agent] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self._obs_dims[agent],),
                dtype=np.float32,
            )
        return self._observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Box:
        """
        Box matching the actor output domain the ASSUME strategies expect.

        ASSUME's actor networks end in a ``softsign``, so their actions live in
        ``[-1, 1]``; ``calculate_bids`` scales them by ``max_bid_price``.
        """
        if agent not in self._action_spaces:
            self._action_spaces[agent] = spaces.Box(
                low=self.action_low,
                high=self.action_high,
                shape=(self._act_dims[agent],),
                dtype=np.float32,
            )
        return self._action_spaces[agent]

    # ------------------------------------------------------------------
    # PettingZoo API
    # ------------------------------------------------------------------
    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        """
        Start a new ASSUME episode and return the first market opening.

        ``seed`` seeds the random number generators the simulation can draw
        from. Note that ASSUME's market clearing breaks *price ties* with
        :func:`random.random` (see ``clearing_algorithms/simple.py``), so
        scenarios containing units that submit identical prices are only
        reproducible when a seed is given.
        """
        if seed is not None:
            # the stdlib RNG is the one ASSUME's clearing uses for tie-breaking
            random.seed(seed)
            np.random.seed(seed)
            try:
                import torch as th

                th.manual_seed(seed)
            except ImportError:  # pragma: no cover - torch is a learning extra
                pass

        self.coordinator.abort()
        self._episode += 1
        self._steps = 0
        self._blocked = None
        self._cycle_key = None
        self._roster = {}

        self.coordinator.start(self._run_episode)

        request = self.coordinator.next_request()
        if request is None:
            raise RuntimeError(
                "The ASSUME simulation finished without asking any unit for an "
                "action. Check that the simulation horizon covers at least one "
                "market opening."
            )

        self._roster = self._build_roster()
        observations = self._collect_cycle(request)

        self.agents = list(self.possible_agents)
        infos = {agent: dict(self._last_infos.get(agent, {})) for agent in self.agents}
        return observations, infos

    def step(
        self, actions: dict[str, np.ndarray]
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict],
    ]:
        """
        Submit one bid per agent, clear the market, and return the next opening.
        """
        if not self.agents:
            raise RuntimeError("step() called on a finished episode - call reset().")

        missing = set(self.agents) - set(actions)
        if missing:
            raise ValueError(f"Missing actions for agents: {sorted(missing)}")

        finished = self._deliver_cycle(actions)
        self._steps += 1

        rewards, reward_infos = self._collect_rewards()

        truncated = (
            self.max_episode_steps is not None and self._steps >= self.max_episode_steps
        )

        if finished or truncated:
            if truncated and not finished:
                self.coordinator.abort()
            observations = {
                agent: np.zeros(self._obs_dims[agent], dtype=np.float32)
                for agent in self.agents
            }
            terminations = {agent: not truncated for agent in self.agents}
            truncations = {agent: truncated for agent in self.agents}
            infos = self._merge_infos(reward_infos)
            self.agents = []
            return observations, rewards, terminations, truncations, infos

        # rebuilds self._last_infos to describe the market opening the returned
        # observations belong to
        observations = self._collect_cycle(self._blocked)
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = self._merge_infos(reward_infos)
        return observations, rewards, terminations, truncations, infos

    def _merge_infos(self, reward_infos: dict[str, dict]) -> dict[str, dict]:
        """
        Build the per-agent info dict.

        ``market_id`` / ``product_start`` / ``product_end`` describe the market
        opening the *returned observation* belongs to, ``reward_*`` describe the
        reward returned alongside it.
        """
        return {
            agent: {
                **self._last_infos.get(agent, {}),
                **reward_infos.get(agent, {}),
            }
            for agent in self.agents
        }

    def render(self) -> None:
        if self.render_mode != "human":
            return
        market_id, start = self._cycle_key or (None, None)
        print(
            f"[assume] episode {self._episode} step {self._steps} {market_id} {start}"
        )

    def close(self) -> None:
        """Abort a running episode and release the simulation thread."""
        self.coordinator.abort()
        self.agents = []

    # ------------------------------------------------------------------
    # cycle handling
    # ------------------------------------------------------------------
    def _collect_cycle(self, request: ActionRequest) -> dict[str, np.ndarray]:
        """
        Turn one blocked simulation request into a full simultaneous-move
        observation batch.

        .. _simultaneity:

        The simulation only asks one unit at a time, because ``calculate_bids``
        is a synchronous call inside mango's event loop. The observations of the
        remaining units of the same market opening are therefore computed here,
        by calling their unmodified ``create_observation``. That is sound because
        an ASSUME observation is a pure read of forecast data and of the unit's
        own past dispatch - no unit state changes while the other units of the
        same opening formulate their bids. The assumption is *verified* rather
        than trusted: :meth:`EnvCoordinator._check_eager_observation` compares
        each pre-computed vector against the one the simulation builds when that
        unit's turn actually arrives.
        """
        self._blocked = request
        self._cycle_key = (request.market_id, request.product_start)
        self.coordinator.clear_eager_observations()

        observations = {request.agent_id: request.observation}

        for agent_id in self.possible_agents:
            if agent_id == request.agent_id:
                continue
            entry = self._roster.get(agent_id)
            if entry is None:
                observations[agent_id] = np.zeros(
                    self._obs_dims[agent_id], dtype=np.float32
                )
                continue
            unit, strategy = entry
            observation = (
                strategy.create_observation(
                    unit=unit,
                    market_id=request.market_id,
                    start=request.product_start,
                    end=request.product_end,
                )
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            observations[agent_id] = observation
            self.coordinator.set_eager_observation(agent_id, observation)

        self._last_infos = {
            agent_id: {
                "market_id": request.market_id,
                "product_start": request.product_start,
                "product_end": request.product_end,
                # NOT "episode": that key is reserved by the gymnasium/SB3
                # Monitor convention for a {"r": ..., "l": ...} stats dict,
                # and frameworks crash if it holds anything else.
                "assume_episode": self._episode,
            }
            for agent_id in self.possible_agents
        }
        return observations

    def _deliver_cycle(self, actions: dict[str, np.ndarray]) -> bool:
        """
        Feed the agents' actions to the simulation until the market opening is
        done and the simulation blocks on the next one.

        Returns:
            True if the simulation reached the end of its horizon.
        """
        served: set[str] = set()
        request = self._blocked

        while request is not None:
            action = actions.get(request.agent_id)
            if action is None:
                raise ValueError(
                    f"The simulation asked agent '{request.agent_id}' for an "
                    "action but none was supplied."
                )
            if request.agent_id in served:
                logger.debug(
                    "Agent %s was asked twice within one market opening (%s, %s); "
                    "reusing the supplied action.",
                    request.agent_id,
                    request.market_id,
                    request.product_start,
                )
            served.add(request.agent_id)

            self.coordinator.send_action(np.asarray(action, dtype=np.float32))

            request = self.coordinator.next_request()
            if request is None:
                self._blocked = None
                return True
            if (request.market_id, request.product_start) != self._cycle_key:
                # the simulation moved on to the next market opening
                self._blocked = request
                return False

        return True

    def _collect_rewards(self) -> tuple[dict[str, float], dict[str, dict]]:
        """
        Aggregate the rewards ASSUME produced while the last step advanced.

        ASSUME computes a reward when the market clears, which happens after all
        bids of an opening have been submitted. Any reward reported while the
        step ran is attributed to that step. The delivery product the reward
        actually belongs to is reported in ``info['reward_product_start']`` so
        the alignment can be verified from the outside.
        """
        rewards = {agent: 0.0 for agent in self.agents}
        infos: dict[str, dict] = {}
        for report in self.coordinator.drain_rewards():
            if report.agent_id not in rewards:
                continue
            rewards[report.agent_id] += report.reward
            info = infos.setdefault(report.agent_id, {})
            info["reward_product_start"] = report.product_start
            info["regret"] = info.get("regret", 0.0) + report.regret
            info["profit"] = info.get("profit", 0.0) + report.profit
        return rewards, infos

    # ------------------------------------------------------------------
    # convenience
    # ------------------------------------------------------------------
    @property
    def observation_mismatches(self) -> int:
        """
        How often a pre-computed observation disagreed with the one the
        simulation built. Should stay at zero; see :meth:`_collect_cycle`.
        """
        return self.coordinator.observation_mismatches

    def __del__(self):  # pragma: no cover - best effort cleanup
        try:
            self.coordinator.abort(join_timeout=5)
        except Exception:
            pass


def parallel_env(**kwargs) -> AssumeParallelEnv:
    """PettingZoo-style factory for :class:`AssumeParallelEnv`."""
    return AssumeParallelEnv(**kwargs)


def raw_parallel_env(**kwargs) -> AssumeParallelEnv:
    """Alias kept for symmetry with the PettingZoo naming conventions."""
    return AssumeParallelEnv(**kwargs)


__all__ = ["AssumeParallelEnv", "parallel_env", "raw_parallel_env"]
