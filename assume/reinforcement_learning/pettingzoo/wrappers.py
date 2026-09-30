# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Optional PettingZoo wrappers for the ASSUME environments.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import numpy as np
from pettingzoo.utils.wrappers import BaseParallelWrapper

logger = logging.getLogger(__name__)


class MarkupActionWrapper(BaseParallelWrapper):
    """
    Reinterpret the action as a **markup on marginal cost** rather than an
    absolute bid price.

    Why you probably want this
    --------------------------
    In the raw action space, action ``0`` means bidding 0 EUR/MWh - far below
    marginal cost, so the unit is dispatched at a loss every hour. A freshly
    initialised policy starts there and has to discover the profitable band
    unaided, which costs a generic RL algorithm a great many episodes.

    ASSUME's own MATD3 does not ask its agents to do that either: during initial
    exploration :meth:`EnergyLearningStrategy.get_actions` adds the unit's
    marginal cost to the action, centring exploration on break-even. This wrapper
    gives an external algorithm the same head start, which also makes it the fair
    comparison point against ASSUME's built-in learner.

    It is additionally the usual formulation in electricity-market RL: a bidding
    policy chooses a markup, not an absolute price.

    How it works
    ------------
    ``bid = clip(marginal_cost + span * action, action_low, action_high)``

    Marginal cost is read from the **last element of the observation vector** -
    the same element ASSUME's own strategy reads. This holds for the shipped
    power-plant and renewable strategies. It does **not** hold for
    :class:`StorageEnergyLearningStrategy`, whose last observation is the cost of
    stored energy; the wrapper refuses to wrap an environment whose agents do not
    expose marginal cost last, unless ``force=True``.

    Args:
        env: The :class:`~assume.reinforcement_learning.pettingzoo.parallel_env.AssumeParallelEnv`
            (or another wrapper around it).
        span: Action ``+-1`` maps to a markup of ``+-span * max_bid_price``.
        force: Skip the strategy check.
    """

    #: Strategies whose observation vector ends with the scaled marginal cost.
    _MARGINAL_COST_LAST = (
        "ExternalEnergyLearningStrategy",
        "ExternalEnergyLearningSingleBidStrategy",
        "ExternalRenewableEnergyLearningSingleBidStrategy",
    )

    def __init__(self, env, span: float = 0.5, force: bool = False):
        super().__init__(env)
        self.span = float(span)
        self._last_observations: dict[str, np.ndarray] = {}
        if not force:
            self._check_strategies()

    def _check_strategies(self) -> None:
        world = getattr(self.unwrapped, "world", None)
        if world is None:  # pragma: no cover - non-ASSUME env
            return
        offenders = {}
        for agent in self.unwrapped.possible_agents:
            unit = world.units.get(agent)
            if unit is None:
                continue
            for strategy in unit.bidding_strategies.values():
                name = type(strategy).__name__
                if name.startswith("External") and name not in self._MARGINAL_COST_LAST:
                    offenders[agent] = name
        if offenders:
            raise ValueError(
                "MarkupActionWrapper assumes the last observation element is the "
                f"unit's scaled marginal cost, which does not hold for {offenders}. "
                "Pass force=True if you know the observation layout is compatible."
            )

    def marginal_cost(self, agent: str) -> float:
        """Scaled marginal cost the wrapper will centre this agent's bid on."""
        return float(self._last_observations[agent][-1])

    def to_price_units(self, action) -> np.ndarray:
        """Convert a policy action to the markup it represents, in price units."""
        return np.asarray(action) * self.span

    def reset(self, seed=None, options=None):
        observations, infos = self.env.reset(seed=seed, options=options)
        self._last_observations = observations
        return observations, infos

    def step(self, actions):
        bids = {}
        for agent, action in actions.items():
            space = self.action_space(agent)
            bids[agent] = np.clip(
                self.marginal_cost(agent) + self.span * np.asarray(action),
                space.low,
                space.high,
            ).astype(np.float32)

        observations, rewards, terminations, truncations, infos = self.env.step(bids)
        self._last_observations = observations
        return observations, rewards, terminations, truncations, infos


class NumericInfoWrapper(BaseParallelWrapper):
    """
    Drop non-numeric entries from ``info``.

    The ASSUME environments report useful diagnostics in ``info`` -
    ``market_id`` (a string), ``product_start`` / ``reward_product_start``
    (datetimes), ``regret``, ``profit``. Most frameworks pass ``info`` through
    untouched, but some build tensor specs from it and choke on anything that is
    not a number. TorchRL's ``PettingZooWrapper`` is one: it calls
    ``torch.as_tensor`` on every info value during ``_init_env`` and raises
    ``TypeError: new(): invalid data type 'str'``.

    Wrap the environment in this to keep only the numeric fields::

        env = NumericInfoWrapper(AssumeParallelEnv(...))

    Args:
        env: The environment to wrap.
        keep: Info keys to keep if their value is numeric. ``None`` keeps every
            numeric value; pass an empty collection to drop ``info`` entirely.
    """

    def __init__(self, env, keep: Iterable[str] | None = None):
        super().__init__(env)
        self.keep = None if keep is None else set(keep)

    def _filter(self, infos: dict) -> dict:
        filtered = {}
        for agent, info in infos.items():
            kept = {}
            for key, value in info.items():
                if self.keep is not None and key not in self.keep:
                    continue
                if isinstance(value, bool | int | float | np.number):
                    kept[key] = float(value)
                elif isinstance(value, np.ndarray) and value.dtype.kind in "fiub":
                    kept[key] = value
            filtered[agent] = kept
        return filtered

    def reset(self, seed=None, options=None):
        observations, infos = self.env.reset(seed=seed, options=options)
        return observations, self._filter(infos)

    def step(self, actions):
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        return observations, rewards, terminations, truncations, self._filter(infos)
