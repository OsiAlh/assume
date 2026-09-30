# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Rendezvous layer between an ASSUME simulation and an external reinforcement
learning loop.

Why this exists
---------------
ASSUME uses *inversion of control*: the simulation drives the policy. A market
opening triggers :meth:`UnitsOperator.submit_bids`, which synchronously calls
``strategy.calculate_bids(...)``, which in turn calls ``strategy.get_actions(...)``.
The policy is never in charge of the clock.

PettingZoo (and every RL framework built on it) requires the opposite: the
trainer drives the environment via ``env.reset()`` / ``env.step(actions)``.

:class:`EnvCoordinator` bridges the two by running the untouched ASSUME
simulation on a **worker thread** and turning ``get_actions`` into a blocking
rendezvous:

1. the simulation thread calls :meth:`request_action` and blocks,
2. the environment thread picks the request up with :meth:`next_request`,
3. the environment thread answers with :meth:`send_action`,
4. the simulation thread resumes and formulates the bid exactly as before.

Only one of the two threads is ever runnable at a time, so no ASSUME state is
ever touched concurrently. The simulation logic itself is completely unchanged.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class EpisodeAborted(Exception):
    """
    Raised inside the simulation thread when the environment is closed or reset
    while the simulation is still waiting for an action.

    It propagates out of ``calculate_bids``. mango logs the failing task and the
    simulation runs to its end without producing further action requests, after
    which the worker thread terminates.
    """


class SimulationFailed(RuntimeError):
    """Raised on the environment thread when the simulation thread died."""


@dataclass
class ActionRequest:
    """A single decision point of one ASSUME unit."""

    agent_id: str
    """The ``unit_id`` of the requesting unit."""

    observation: np.ndarray
    """Observation as built by ASSUME's unmodified ``create_observation``."""

    market_id: str | None
    """Market the bid is being formulated for."""

    product_start: datetime | None
    """Delivery start of the first product in this opening."""

    product_end: datetime | None
    """Delivery end of the product window passed to ``create_observation``."""

    eager: bool = False
    """True if the observation was materialised ahead of the simulation."""


@dataclass
class RewardReport:
    """A reward as computed by ASSUME's unmodified ``calculate_reward``."""

    agent_id: str
    product_start: datetime
    reward: float
    regret: float
    profit: float


@dataclass
class _Sentinel:
    kind: str


_EPISODE_END = _Sentinel("episode_end")
_ABORT = _Sentinel("abort")


class _AbortNoiseFilter(logging.Filter):
    """
    Drops mango's ERROR traceback for a deliberate abort.

    Aborting an episode raises :class:`EpisodeAborted` inside ``calculate_bids``,
    which mango's scheduler faithfully logs as a failed task. That is expected,
    not an error, so it is filtered out - genuine failures still get through.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        exc_info = record.exc_info
        return not (exc_info and isinstance(exc_info[1], EpisodeAborted))


def _silence_abort_tracebacks() -> None:
    """Install :class:`_AbortNoiseFilter` on mango's scheduler logger, once."""
    mango_logger = logging.getLogger("mango.util.scheduling")
    if not any(isinstance(f, _AbortNoiseFilter) for f in mango_logger.filters):
        mango_logger.addFilter(_AbortNoiseFilter())


class EnvCoordinator:
    """
    Thread-safe hand-off between the ASSUME simulation thread and the RL loop.

    The queues have ``maxsize=1`` on purpose: they enforce a strict ping-pong so
    that the simulation thread is suspended whenever the environment thread runs
    and vice versa.

    Args:
        name: Used for thread naming and log messages.
    """

    def __init__(self, name: str = "assume-sim"):
        self.name = name
        _silence_abort_tracebacks()

        self._request_q: queue.Queue = queue.Queue(maxsize=1)
        self._action_q: queue.Queue = queue.Queue(maxsize=1)

        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._aborted = threading.Event()
        self._finished = threading.Event()

        # rewards are written by the simulation thread and drained by the env
        # thread; both never run at the same time, the lock is belt-and-braces
        self._reward_lock = threading.Lock()
        self._rewards: list[RewardReport] = []

        # observations pre-computed by the environment for agents that the
        # simulation has not asked yet in the current decision cycle
        self._eager_observations: dict[str, np.ndarray] = {}
        self.observation_mismatches: int = 0
        self.strict_observations: bool = False

    # ------------------------------------------------------------------
    # lifecycle (environment thread)
    # ------------------------------------------------------------------
    def start(self, target: Callable[[], Any]) -> None:
        """
        Start the simulation on a worker thread.

        Args:
            target: Callable that performs one full ASSUME episode
                (typically ``setup_world`` followed by ``world.run()``).
        """
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("An episode is still running - call close() first.")

        self._request_q = queue.Queue(maxsize=1)
        self._action_q = queue.Queue(maxsize=1)
        self._error = None
        self._aborted.clear()
        self._finished.clear()
        self._rewards.clear()
        self._eager_observations.clear()

        def _runner() -> None:
            try:
                target()
            except EpisodeAborted:
                logger.debug("%s: simulation aborted on request", self.name)
            except BaseException as exc:  # noqa: BLE001 - forwarded to env thread
                logger.exception("%s: simulation thread failed", self.name)
                self._error = exc
            finally:
                self._finished.set()
                # unblock a possibly waiting environment thread
                try:
                    self._request_q.put(_EPISODE_END, timeout=5)
                except queue.Full:  # pragma: no cover - env is not listening
                    pass

        self._thread = threading.Thread(
            target=_runner, name=f"{self.name}-worker", daemon=True
        )
        self._thread.start()

    def next_request(self, timeout: float | None = None) -> ActionRequest | None:
        """
        Block until the simulation asks for the next action.

        Returns:
            The pending :class:`ActionRequest`, or ``None`` when the episode
            ended (the simulation reached its end date or died).

        Raises:
            SimulationFailed: if the simulation thread raised.
        """
        item = self._request_q.get(timeout=timeout)
        if isinstance(item, _Sentinel):
            if self._error is not None:
                raise SimulationFailed(
                    f"ASSUME simulation failed: {self._error!r}"
                ) from self._error
            return None
        return item

    def send_action(self, action: np.ndarray) -> None:
        """Hand an action to the simulation thread, unblocking it."""
        self._action_q.put(action)

    def abort(self, join_timeout: float = 30.0) -> None:
        """
        Tell the simulation thread to give up and wait for it to finish.

        Safe to call multiple times and when no episode is running.
        """
        if self._thread is None:
            return
        self._aborted.set()

        # Drain a pending request and release the simulation thread. The loop is
        # bounded: during interpreter shutdown a daemon thread can stop making
        # progress entirely, and without a deadline close() would spin forever.
        deadline = time.monotonic() + join_timeout
        while self._thread.is_alive() and not self._finished.is_set():
            if time.monotonic() >= deadline:
                logger.warning(
                    "%s: simulation thread did not react to the abort within "
                    "%ss; abandoning it (it is a daemon thread).",
                    self.name,
                    join_timeout,
                )
                self._thread = None
                return
            try:
                self._action_q.put(_ABORT, timeout=0.1)
            except queue.Full:
                pass
            try:
                self._request_q.get(timeout=0.1)
            except queue.Empty:
                pass
        self._thread.join(timeout=max(0.0, deadline - time.monotonic()))
        if self._thread.is_alive():  # pragma: no cover - defensive
            logger.warning(
                "%s: simulation thread did not stop within %ss; it is a daemon "
                "thread and will be killed on interpreter exit.",
                self.name,
                join_timeout,
            )
        self._thread = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # eager observations (environment thread)
    # ------------------------------------------------------------------
    def set_eager_observation(self, agent_id: str, observation: np.ndarray) -> None:
        """
        Register an observation that the environment computed ahead of time.

        Used by :class:`~assume.reinforcement_learning.pettingzoo.parallel_env.AssumeParallelEnv`
        to build a simultaneous-move view of one market opening. When the
        simulation later asks that agent for an action, the observation it
        builds is compared against this one - see :meth:`request_action`.
        """
        self._eager_observations[agent_id] = np.asarray(observation, dtype=np.float32)

    def clear_eager_observations(self) -> None:
        self._eager_observations.clear()

    # ------------------------------------------------------------------
    # simulation side (simulation thread)
    # ------------------------------------------------------------------
    def request_action(
        self,
        agent_id: str,
        observation: np.ndarray,
        market_id: str | None = None,
        product_start: datetime | None = None,
        product_end: datetime | None = None,
    ) -> np.ndarray:
        """
        Publish an observation and block until the environment supplies an action.

        Called from the simulation thread by
        :class:`~assume.reinforcement_learning.pettingzoo.strategies.ExternalControlMixin`.

        Raises:
            EpisodeAborted: when the environment was closed or reset meanwhile.
        """
        if self._aborted.is_set():
            raise EpisodeAborted(f"{self.name}: episode aborted before {agent_id}")

        observation = np.asarray(observation, dtype=np.float32)
        self._check_eager_observation(agent_id, observation)

        self._request_q.put(
            ActionRequest(
                agent_id=agent_id,
                observation=observation,
                market_id=market_id,
                product_start=product_start,
                product_end=product_end,
            )
        )
        action = self._action_q.get()
        if isinstance(action, _Sentinel):
            raise EpisodeAborted(
                f"{self.name}: episode aborted while {agent_id} waited"
            )
        return action

    def report_reward(
        self,
        agent_id: str,
        product_start: datetime,
        reward: float,
        regret: float = 0.0,
        profit: float = 0.0,
    ) -> None:
        """
        Record a reward produced by ASSUME's unmodified ``calculate_reward``.

        Called from the simulation thread after market clearing.
        """
        with self._reward_lock:
            self._rewards.append(
                RewardReport(
                    agent_id=agent_id,
                    product_start=product_start,
                    reward=float(reward),
                    regret=float(regret),
                    profit=float(profit),
                )
            )

    def drain_rewards(self) -> list[RewardReport]:
        """Take and clear every reward reported since the last call."""
        with self._reward_lock:
            reports = self._rewards
            self._rewards = []
        return reports

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _check_eager_observation(self, agent_id: str, observation: np.ndarray) -> None:
        """
        Validate the assumption behind eager observation materialisation.

        Observations in ASSUME are pure reads of forecast data and the unit's own
        past dispatch; no unit state changes while the other units of the same
        market opening formulate their bids. This method verifies that at
        runtime instead of trusting it.
        """
        expected = self._eager_observations.pop(agent_id, None)
        if expected is None:
            return
        if expected.shape != observation.shape or not np.allclose(
            expected, observation, atol=1e-6, rtol=0.0
        ):
            self.observation_mismatches += 1
            message = (
                f"{self.name}: the observation pre-computed for '{agent_id}' differs "
                f"from the one the simulation built at its turn. The action was "
                f"selected from the pre-computed observation. This means an "
                f"observation of this strategy is not independent of the other "
                f"agents' bids within the same market opening - use the AEC "
                f"environment instead of the parallel one."
            )
            if self.strict_observations:
                raise RuntimeError(message)
            logger.warning(message)


@dataclass
class CoordinatorHandle:
    """
    Small holder passed to strategies so they can find their coordinator.

    Strategies are constructed deep inside ``setup_world`` where the environment
    has no reference to them, so the handle is injected through the learning
    role instead.
    """

    coordinator: EnvCoordinator
    agent_ids: set[str] = field(default_factory=set)
