# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
PettingZoo ``AECEnv`` view of an ASSUME market simulation.

A market opening is a simultaneous-move game: every unit submits its bid without
seeing the others' bids. The native representation of that is therefore the
:class:`~assume.reinforcement_learning.pettingzoo.parallel_env.AssumeParallelEnv`,
and the AEC view is derived from it with PettingZoo's official
:func:`~pettingzoo.utils.conversions.parallel_to_aec` conversion, which buffers
the agents' actions and applies them all at once.

Use this module when a framework expects the AEC API (Tianshou's
``PettingZooEnv``, RLlib's ``PettingZooEnv``, PettingZoo's own ``api_test``).
"""

from __future__ import annotations

from pettingzoo.utils import wrappers
from pettingzoo.utils.conversions import parallel_to_aec

from assume.reinforcement_learning.pettingzoo.parallel_env import AssumeParallelEnv


def raw_env(**kwargs):
    """AEC environment without the safety wrappers."""
    return parallel_to_aec(AssumeParallelEnv(**kwargs))


def env(**kwargs):
    """
    AEC environment with PettingZoo's standard safety wrappers.

    Adds :class:`~pettingzoo.utils.wrappers.AssertOutOfBoundsWrapper` (rejects
    actions outside the action space) and
    :class:`~pettingzoo.utils.wrappers.OrderEnforcingWrapper` (rejects calls made
    out of order).

    Takes the same keyword arguments as
    :class:`~assume.reinforcement_learning.pettingzoo.parallel_env.AssumeParallelEnv`.
    """
    aec_env = raw_env(**kwargs)
    aec_env = wrappers.AssertOutOfBoundsWrapper(aec_env)
    aec_env = wrappers.OrderEnforcingWrapper(aec_env)
    return aec_env


#: Backwards-compatible alias; ``AssumeAECEnv(...)`` builds the wrapped AEC env.
AssumeAECEnv = env

__all__ = ["env", "raw_env", "AssumeAECEnv"]
