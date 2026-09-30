# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
PettingZoo bindings for ASSUME.

They let any multi-agent RL framework that speaks PettingZoo (RLlib, Tianshou,
SuperSuit + Stable-Baselines3, CleanRL, MARLlib, ...) train bidding policies on
an ASSUME electricity-market simulation.

The market model is untouched
-----------------------------
Nothing about how ASSUME simulates markets changes. Observations, the mapping
from action to orderbook, market clearing and the reward function are the ones
ASSUME already ships; the bindings only redirect *where the action comes from*.
Concretely, an externally controlled strategy is a subclass of the corresponding
ASSUME learning strategy that overrides a single method, ``get_actions``.

How control is inverted
-----------------------
ASSUME drives the policy: a market opening calls ``calculate_bids`` which calls
``get_actions``. PettingZoo requires the policy to drive the environment. The
simulation therefore runs on a worker thread and ``get_actions`` blocks on a
rendezvous with the environment thread
(:class:`~assume.reinforcement_learning.pettingzoo.env_coordinator.EnvCoordinator`).
Only one of the two threads is runnable at any moment, so no ASSUME state is
ever touched concurrently.

Quick start
-----------
::

    from assume.reinforcement_learning.pettingzoo import AssumeParallelEnv

    env = AssumeParallelEnv(
        inputs_path="examples/inputs",
        scenario="example_02a",
        study_case="tiny",
    )
    observations, infos = env.reset(seed=0)
    while env.agents:
        actions = {a: env.action_space(a).sample() for a in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
    env.close()

For the AEC API::

    from assume.reinforcement_learning.pettingzoo import aec_env
    env = aec_env.env(inputs_path=..., scenario=..., study_case=...)

To let an external algorithm bid a *markup on marginal cost* instead of an
absolute price - the same head start ASSUME's own MATD3 gives itself::

    from assume.reinforcement_learning.pettingzoo import MarkupActionWrapper
    env = MarkupActionWrapper(AssumeParallelEnv(...))
"""

from assume.reinforcement_learning.pettingzoo.env_coordinator import (
    ActionRequest,
    EnvCoordinator,
    EpisodeAborted,
    RewardReport,
    SimulationFailed,
)
from assume.reinforcement_learning.pettingzoo.external_learning_role import (
    ExternalLearning,
)
from assume.reinforcement_learning.pettingzoo.parallel_env import (
    AssumeParallelEnv,
    parallel_env,
)
from assume.reinforcement_learning.pettingzoo.strategies import (
    EXTERNAL_STRATEGY_ALIASES,
    EXTERNAL_STRATEGY_NAMES,
    ExternalControlMixin,
    ExternalEnergyLearningSingleBidStrategy,
    ExternalEnergyLearningStrategy,
    ExternalRenewableEnergyLearningSingleBidStrategy,
    ExternalStorageEnergyLearningStrategy,
)
from assume.reinforcement_learning.pettingzoo.wrappers import (
    MarkupActionWrapper,
    NumericInfoWrapper,
)
from assume.reinforcement_learning.pettingzoo import aec_env

__all__ = [
    "AssumeParallelEnv",
    "parallel_env",
    "aec_env",
    "MarkupActionWrapper",
    "NumericInfoWrapper",
    "EnvCoordinator",
    "ActionRequest",
    "RewardReport",
    "EpisodeAborted",
    "SimulationFailed",
    "ExternalLearning",
    "ExternalControlMixin",
    "ExternalEnergyLearningStrategy",
    "ExternalEnergyLearningSingleBidStrategy",
    "ExternalStorageEnergyLearningStrategy",
    "ExternalRenewableEnergyLearningSingleBidStrategy",
    "EXTERNAL_STRATEGY_ALIASES",
    "EXTERNAL_STRATEGY_NAMES",
]
