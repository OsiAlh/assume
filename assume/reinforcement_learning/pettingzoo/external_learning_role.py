# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Learning role used when actions come from an external RL framework.

:class:`ExternalLearning` is a thin subclass of ASSUME's :class:`Learning` role.
It keeps every piece of ASSUME's bookkeeping (observation/action/reward caches,
``rl_params`` database export, TensorBoard hooks) and only *additionally*
forwards rewards to an :class:`EnvCoordinator`.

The role is always constructed with ``learning_mode=True`` **and**
``evaluation_mode=True``:

* ``learning_mode=True`` makes the unmodified strategies populate the
  observation / action / reward caches, which is how the reward reaches us.
* ``evaluation_mode=True`` makes :meth:`Learning.store_to_buffer_and_update`
  skip the replay buffer and the MATD3 policy update - training is the external
  framework's job, not ASSUME's.
"""

from __future__ import annotations

import logging
from datetime import datetime

from assume.common.base import LearningConfig
from assume.reinforcement_learning.learning_role import Learning
from assume.reinforcement_learning.pettingzoo.env_coordinator import EnvCoordinator

logger = logging.getLogger(__name__)


class ExternalLearning(Learning):
    """
    ASSUME learning role whose experience is streamed to an external RL loop.

    Args:
        learning_config: Same configuration object ASSUME always uses.
        start: Simulation start.
        end: Simulation end.
        coordinator: The rendezvous object shared with the PettingZoo env.
    """

    def __init__(
        self,
        learning_config: LearningConfig,
        start: datetime,
        end: datetime,
        coordinator: EnvCoordinator,
    ):
        super().__init__(learning_config=learning_config, start=start, end=end)
        self.coordinator = coordinator

    def add_observation_to_cache(self, unit_id, start, observation) -> None:
        """
        Store the observation, replacing an earlier entry for the same
        ``(start, unit_id)`` instead of appending a second one.

        The parallel environment pre-computes observations (see
        :meth:`EnvCoordinator.set_eager_observation`), so ``create_observation``
        can legitimately run twice for the same decision point. ASSUME's own
        consumers only ever read element ``[0]`` of these lists, so replacing
        keeps the cache identical to a plain ASSUME run.
        """
        self.all_obs[start][unit_id] = [observation]

    def write_rl_params_to_output(self, cache) -> None:
        """
        Skip the export when the cache holds no completed time step.

        ``Learning.write_rl_params_to_output`` does ``next(iter(cache["obs"]))``
        and raises ``StopIteration`` on an empty cache. That happens whenever a
        scheduled export finds only time steps whose reward has not been
        computed yet - routine here, because closing the environment aborts the
        episode while bids are in flight. This is an upstream robustness gap;
        guarding it here keeps ASSUME's own code untouched.
        """
        if not cache.get("obs"):
            return
        super().write_rl_params_to_output(cache)

    def add_reward_to_cache(self, unit_id, start, reward, regret, profit) -> None:
        """Keep ASSUME's caching behaviour and additionally publish the reward."""
        super().add_reward_to_cache(unit_id, start, reward, regret, profit)

        self.coordinator.report_reward(
            agent_id=unit_id,
            product_start=start,
            reward=float(reward),
            regret=float(regret),
            profit=float(profit),
        )
