# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Bidding strategies whose actions come from an external RL framework.

Each class here is a *two-method* subclass of one of ASSUME's shipped learning
strategies:

``create_observation``
    remembers which market and delivery period the current call belongs to, then
    delegates to ASSUME unchanged.

``get_actions``
    hands the observation to the :class:`EnvCoordinator` and blocks until the
    PettingZoo environment answers, instead of querying the local actor network.

Everything else - how the observation vector is assembled, how an action is
turned into an orderbook, how the reward is computed - is inherited verbatim.
That is the whole point: the market model and the RL interface stay identical to
a normal ASSUME run, only the source of the action changes.
"""

from __future__ import annotations

import logging

import numpy as np
import torch as th

from assume.common.base import BaseUnit
from assume.strategies.learning_strategies import (
    EnergyLearningSingleBidStrategy,
    EnergyLearningStrategy,
    RenewableEnergyLearningSingleBidStrategy,
    StorageEnergyLearningStrategy,
)

logger = logging.getLogger(__name__)


class ExternalControlMixin:
    """
    Replaces the actor network with a blocking call to the environment.

    Must be mixed in **before** the ASSUME strategy so that its ``get_actions``
    wins the MRO, e.g.::

        class MyExternal(ExternalControlMixin, EnergyLearningStrategy):
            pass
    """

    #: filled by ``create_observation`` so ``get_actions`` knows its context
    _external_context: tuple[str | None, object, object] = (None, None, None)

    @property
    def coordinator(self):
        """The :class:`EnvCoordinator` injected through the learning role."""
        coordinator = getattr(self.learning_role, "coordinator", None)
        if coordinator is None:
            raise RuntimeError(
                f"{type(self).__name__} for unit '{self.unit_id}' is externally "
                "controlled but no EnvCoordinator is attached to the learning "
                "role. Build the world through one of the PettingZoo "
                "environments in assume.reinforcement_learning.pettingzoo."
            )
        return coordinator

    def create_observation(self, unit: BaseUnit, market_id: str, start, end):  # noqa: D102 - inherited docstring
        self._external_context = (market_id, start, end)
        return super().create_observation(
            unit=unit, market_id=market_id, start=start, end=end
        )

    def get_actions(self, next_observation):
        """
        Ask the external policy for an action.

        Blocks the simulation thread until the PettingZoo environment supplies
        one. The returned noise tensor is all zeros: exploration is the external
        framework's responsibility, so nothing is added on top of the action.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: action and (zero) noise, in the
            same dtype/device the ASSUME strategies expect.
        """
        market_id, start, end = self._external_context

        observation = next_observation.detach().cpu().numpy().astype(np.float32)

        action = self.coordinator.request_action(
            agent_id=self.unit_id,
            observation=observation,
            market_id=market_id,
            product_start=start,
            product_end=end,
        )

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != self.act_dim:
            raise ValueError(
                f"Agent '{self.unit_id}' expects an action of dimension "
                f"{self.act_dim}, got shape {action.shape}."
            )

        curr_action = th.as_tensor(action, dtype=self.float_type, device=self.device)
        noise = th.zeros_like(curr_action)

        return curr_action, noise


class ExternalEnergyLearningStrategy(ExternalControlMixin, EnergyLearningStrategy):
    """Externally controlled variant of :class:`EnergyLearningStrategy`."""


class ExternalEnergyLearningSingleBidStrategy(
    ExternalControlMixin, EnergyLearningSingleBidStrategy
):
    """Externally controlled variant of :class:`EnergyLearningSingleBidStrategy`."""


class ExternalStorageEnergyLearningStrategy(
    ExternalControlMixin, StorageEnergyLearningStrategy
):
    """Externally controlled variant of :class:`StorageEnergyLearningStrategy`."""


class ExternalRenewableEnergyLearningSingleBidStrategy(
    ExternalControlMixin, RenewableEnergyLearningSingleBidStrategy
):
    """Externally controlled variant of :class:`RenewableEnergyLearningSingleBidStrategy`."""


#: Maps the ``bidding_strategies`` key used in ``config.yaml`` / the unit CSVs to
#: its externally controlled counterpart. The PettingZoo environments apply this
#: mapping to ``world.bidding_strategies`` so existing scenarios work unchanged.
EXTERNAL_STRATEGY_ALIASES: dict[str, type] = {
    # current names
    "powerplant_energy_learning": ExternalEnergyLearningStrategy,
    "powerplant_energy_learning_single_bid": ExternalEnergyLearningSingleBidStrategy,
    "storage_energy_learning": ExternalStorageEnergyLearningStrategy,
    "renewable_energy_learning_single_bid": (
        ExternalRenewableEnergyLearningSingleBidStrategy
    ),
    # deprecated names still accepted by ASSUME
    "pp_learning": ExternalEnergyLearningStrategy,
    "storage_learning": ExternalStorageEnergyLearningStrategy,
    "renewable_eom_learning": ExternalRenewableEnergyLearningSingleBidStrategy,
    "learning_advanced_orders": ExternalEnergyLearningStrategy,
}

#: Names under which the external strategies can be requested explicitly from a
#: scenario file, e.g. ``bidding_EOM: powerplant_energy_external``.
EXTERNAL_STRATEGY_NAMES: dict[str, type] = {
    "powerplant_energy_external": ExternalEnergyLearningStrategy,
    "powerplant_energy_external_single_bid": ExternalEnergyLearningSingleBidStrategy,
    "storage_energy_external": ExternalStorageEnergyLearningStrategy,
    "renewable_energy_external_single_bid": (
        ExternalRenewableEnergyLearningSingleBidStrategy
    ),
}
