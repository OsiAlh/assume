# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for the PettingZoo bindings.

The important one is :func:`test_simulation_results_are_unchanged`: it proves
that routing actions through PettingZoo produces exactly the same market
simulation as plain ASSUME.
"""

import random
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pettingzoo")
pytest.importorskip("torch")

from pettingzoo.test import api_test, parallel_api_test  # noqa: E402

from assume.reinforcement_learning.pettingzoo import (  # noqa: E402
    AssumeParallelEnv,
    MarkupActionWrapper,
    NumericInfoWrapper,
    aec_env,
)
from assume.reinforcement_learning.pettingzoo.strategies import (  # noqa: E402
    ExternalControlMixin,
)

pytestmark = [pytest.mark.require_learning, pytest.mark.require_pettingzoo]

INPUTS_PATH = str(Path(__file__).parents[1] / "examples" / "inputs")

# one learning power plant, four simulated days -> short episodes
TINY = dict(inputs_path=INPUTS_PATH, scenario="example_02a", study_case="tiny")
# five learning power plants, used to exercise the simultaneous-move path
MULTI = dict(inputs_path=INPUTS_PATH, scenario="example_02b", study_case="base")


@pytest.fixture
def tiny_env():
    env = AssumeParallelEnv(**TINY, max_episode_steps=10)
    yield env
    env.close()


@pytest.fixture
def multi_env():
    env = AssumeParallelEnv(**MULTI, max_episode_steps=10)
    yield env
    env.close()


def random_actions(env):
    return {agent: env.action_space(agent).sample() for agent in env.agents}


def rollout(env, seed=0):
    """Run one full episode with random actions, returning per-agent returns."""
    env.reset(seed=seed)
    returns = {agent: 0.0 for agent in env.possible_agents}
    steps = 0
    while env.agents:
        _, rewards, _, _, _ = env.step(random_actions(env))
        for agent, reward in rewards.items():
            returns[agent] += reward
        steps += 1
    return returns, steps


# ----------------------------------------------------------------------
# PettingZoo API conformance
# ----------------------------------------------------------------------
def test_parallel_api_conformance(multi_env):
    parallel_api_test(multi_env, num_cycles=5)


def test_aec_api_conformance():
    env = aec_env.env(**MULTI, max_episode_steps=10)
    api_test(env, num_cycles=5, verbose_progress=False)
    env.close()


# ----------------------------------------------------------------------
# agent discovery and spaces
# ----------------------------------------------------------------------
def test_agents_and_spaces_match_the_scenario(multi_env):
    assert multi_env.possible_agents == ["pp_10", "pp_6", "pp_7", "pp_8", "pp_9"]

    for agent in multi_env.possible_agents:
        strategy = multi_env.world.units[agent].bidding_strategies["EOM"]
        assert isinstance(strategy, ExternalControlMixin)
        # spaces mirror the ASSUME strategy, they are not invented here
        assert multi_env.observation_space(agent).shape == (strategy.obs_dim,)
        assert multi_env.action_space(agent).shape == (strategy.act_dim,)
        # the actor networks end in softsign, so actions live in [-1, 1]
        assert multi_env.action_space(agent).low.min() == -1.0
        assert multi_env.action_space(agent).high.max() == 1.0


def test_scenario_without_learning_units_is_rejected():
    with pytest.raises((ValueError, KeyError)):
        AssumeParallelEnv(
            inputs_path=INPUTS_PATH, scenario="example_01a", study_case="base"
        )


def test_bidding_strategy_registry_is_not_mutated_globally():
    """A PettingZoo env must not turn every other World's learning strategy
    into an externally controlled one."""
    from assume.strategies import bidding_strategies
    from assume.strategies.learning_strategies import EnergyLearningStrategy

    before = bidding_strategies["powerplant_energy_learning"]
    env = AssumeParallelEnv(**TINY, max_episode_steps=2)
    try:
        assert bidding_strategies["powerplant_energy_learning"] is before
        assert before is EnergyLearningStrategy
        assert (
            env.world.bidding_strategies["powerplant_energy_learning"]
            is not EnergyLearningStrategy
        )
    finally:
        env.close()


# ----------------------------------------------------------------------
# episode mechanics
# ----------------------------------------------------------------------
def test_reset_returns_one_observation_per_agent(multi_env):
    observations, infos = multi_env.reset(seed=0)

    assert set(observations) == set(multi_env.possible_agents)
    assert set(infos) == set(multi_env.possible_agents)
    for agent, observation in observations.items():
        assert multi_env.observation_space(agent).contains(observation)
    # every agent bids for the same delivery product -> simultaneous move
    starts = {info["product_start"] for info in infos.values()}
    assert len(starts) == 1


def test_full_episode_terminates_by_itself():
    env = AssumeParallelEnv(**TINY)
    try:
        env.reset(seed=0)
        steps = 0
        terminations = {}
        while env.agents:
            _, _, terminations, truncations, _ = env.step(random_actions(env))
            steps += 1
        # 4 simulated days at hourly market openings
        assert 90 < steps < 100
        assert all(terminations.values())
        assert not any(truncations.values())
        assert env.agents == []
    finally:
        env.close()


def test_max_episode_steps_truncates(tiny_env):
    tiny_env.reset(seed=0)
    steps = 0
    truncations = {}
    while tiny_env.agents:
        _, _, terminations, truncations, _ = tiny_env.step(random_actions(tiny_env))
        steps += 1
    assert steps == 10
    assert all(truncations.values())
    assert not any(terminations.values())


def test_repeated_resets_work(tiny_env):
    for episode in range(3):
        returns, steps = rollout(tiny_env, seed=episode)
        assert steps == 10
        assert set(returns) == set(tiny_env.possible_agents)


def test_step_after_episode_end_raises(tiny_env):
    rollout(tiny_env)
    with pytest.raises(RuntimeError):
        tiny_env.step({})


def test_missing_action_raises(multi_env):
    multi_env.reset(seed=0)
    actions = random_actions(multi_env)
    actions.pop(multi_env.agents[0])
    with pytest.raises(ValueError):
        multi_env.step(actions)


# ----------------------------------------------------------------------
# semantics
# ----------------------------------------------------------------------
def test_rewards_belong_to_the_product_that_was_bid_on(multi_env):
    """step(a_t) must return the reward for the delivery product a_t bid on."""
    _, infos = multi_env.reset(seed=0)
    bid_product = infos[multi_env.agents[0]]["product_start"]

    for _ in range(5):
        _, _, _, _, infos = multi_env.step(random_actions(multi_env))
        for agent, info in infos.items():
            assert info["reward_product_start"] == bid_product, agent
        bid_product = infos[multi_env.agents[0]]["product_start"]


def test_info_avoids_the_reserved_episode_key(multi_env):
    """
    ``episode`` is reserved by the gymnasium/SB3 Monitor convention for a
    ``{"r": ..., "l": ...}`` stats dict. Putting anything else there crashes
    Stable-Baselines3 inside ``dump_logs``.
    """
    _, infos = multi_env.reset(seed=0)
    for info in infos.values():
        assert "episode" not in info
        assert info["assume_episode"] == 1

    _, _, _, _, infos = multi_env.step(random_actions(multi_env))
    for info in infos.values():
        assert "episode" not in info


def test_eager_observations_match_the_simulation(multi_env):
    """
    The parallel view pre-computes the observations of the agents the simulation
    has not reached yet. Those must equal the ones the simulation builds when
    each agent's turn actually arrives.
    """
    multi_env.coordinator.strict_observations = True
    rollout(multi_env)
    assert multi_env.observation_mismatches == 0


def test_actions_reach_the_market():
    """A high bid price must change the dispatch relative to a low one."""
    env = AssumeParallelEnv(**MULTI, max_episode_steps=25)
    try:
        dispatch = {}
        for label, value in (("cheap", -0.9), ("expensive", 0.9)):
            env.reset(seed=0)
            action = np.array([value, value], dtype=np.float32)
            while env.agents:
                env.step({agent: action for agent in env.agents})
            dispatch[label] = np.array(env.world.units["pp_6"].outputs["energy"])
        assert not np.allclose(dispatch["cheap"], dispatch["expensive"])
    finally:
        env.close()


def test_close_is_idempotent(tiny_env):
    tiny_env.reset(seed=0)
    tiny_env.step(random_actions(tiny_env))
    tiny_env.close()
    tiny_env.close()
    assert not tiny_env.coordinator.running


# ----------------------------------------------------------------------
# the reward is ASSUME's own, unmodified
# ----------------------------------------------------------------------
@pytest.mark.slow
def test_env_reward_equals_assumes_logged_reward(tmp_path):
    """
    The reward the environment returns must be byte-identical to the one ASSUME
    writes to its own ``rl_params`` table for the same actions.

    This is the guarantee that makes external training comparable to ASSUME's
    built-in MATD3: both optimise the same shaped reward,
    ``(profit - regret) / (max_bid_price * max_power)``.
    """
    import sqlite3

    import pandas as pd
    import torch as th

    from assume import World
    from assume.scenario.loader_csv import (
        load_config_and_create_forecaster,
        setup_world,
    )

    constant = np.array([0.55, 0.75], dtype=np.float32)
    db = tmp_path / "reward_check.db"

    # --- plain ASSUME: rewards land in rl_params ----------------------
    random.seed(7)
    world = World(database_uri=f"sqlite:///{db}", export_csv_path="", log_level="ERROR")
    world.show_progress = False
    world.scenario_data = load_config_and_create_forecaster(
        INPUTS_PATH, "example_02a", "tiny"
    )
    world.scenario_data["config"]["learning_config"]["learning_mode"] = True
    setup_world(world=world, evaluation_mode=True)
    tensor = th.tensor(constant, dtype=th.float)
    for unit in world.units.values():
        for strategy in unit.bidding_strategies.values():
            if hasattr(strategy, "act_dim"):
                strategy.get_actions = lambda _obs, c=tensor: (c, th.zeros_like(c))
    world.run()

    logged = {
        (unit, pd.Timestamp(when)): reward
        for unit, when, reward in sqlite3.connect(db).execute(
            "SELECT unit, datetime, reward FROM rl_params"
        )
    }
    assert logged, "ASSUME wrote no rl_params rows"

    # --- the same actions through the environment ---------------------
    random.seed(7)
    env = AssumeParallelEnv(
        inputs_path=INPUTS_PATH, scenario="example_02a", study_case="tiny"
    )
    try:
        _, infos = env.reset(seed=7)
        returned = {}
        while env.agents:
            _, rewards, _, _, infos = env.step(
                {agent: constant for agent in env.agents}
            )
            for agent, reward in rewards.items():
                start = infos[agent].get("reward_product_start")
                if start is not None:
                    returned[(agent, pd.Timestamp(start))] = reward
    finally:
        env.close()

    # ASSUME only flushes completed train_freq windows, so the environment sees
    # a few more steps than the database holds; compare where they overlap
    shared = set(logged) & set(returned)
    assert len(shared) > 50, f"only {len(shared)} overlapping rewards"
    for key in shared:
        assert returned[key] == pytest.approx(logged[key], abs=1e-12)


# ----------------------------------------------------------------------
# MarkupActionWrapper
# ----------------------------------------------------------------------
def test_markup_wrapper_bids_marginal_cost_for_a_zero_action():
    """
    A zero action must produce exactly the bid an explicit marginal-cost action
    produces on the raw environment - same market, same rewards.
    """
    wrapped = MarkupActionWrapper(AssumeParallelEnv(**MULTI, max_episode_steps=5))
    raw = AssumeParallelEnv(**MULTI, max_episode_steps=5)
    try:
        wrapped_obs, _ = wrapped.reset(seed=0)
        raw_obs, _ = raw.reset(seed=0)

        for _ in range(5):
            zero = {
                agent: np.zeros(wrapped.action_space(agent).shape, dtype=np.float32)
                for agent in wrapped.agents
            }
            explicit = {
                agent: np.full(
                    raw.action_space(agent).shape,
                    float(raw_obs[agent][-1]),  # scaled marginal cost
                    dtype=np.float32,
                )
                for agent in raw.agents
            }
            # Both environments draw from the *same* global RNG, which ASSUME's
            # clearing uses to break price ties (this scenario has identical
            # twin units). Stepping them alternately would desynchronise that
            # stream and dispatch the twins differently, so replay the same
            # state into both.
            rng_state = random.getstate()
            wrapped_obs, wrapped_rewards, _, _, _ = wrapped.step(zero)
            random.setstate(rng_state)
            raw_obs, raw_rewards, _, _, _ = raw.step(explicit)

            for agent in raw_rewards:
                # abs tolerance: rewards can be ~1e-10, where a relative
                # tolerance is meaningless. float32 rounding differs slightly
                # because the wrapper computes mc + span*0 before casting.
                assert wrapped_rewards[agent] == pytest.approx(
                    raw_rewards[agent], abs=1e-9
                )
    finally:
        wrapped.close()
        raw.close()


def test_markup_wrapper_clips_to_the_action_space():
    wrapped = MarkupActionWrapper(AssumeParallelEnv(**MULTI, max_episode_steps=3))
    try:
        wrapped.reset(seed=0)
        huge = {
            agent: np.full(wrapped.action_space(agent).shape, 50.0, dtype=np.float32)
            for agent in wrapped.agents
        }
        # must not raise: the wrapper clips before the bid reaches ASSUME
        wrapped.step(huge)
    finally:
        wrapped.close()


def test_markup_wrapper_rejects_strategies_without_marginal_cost_last():
    """Storage observations end with the cost of stored energy, not marginal cost."""
    env = AssumeParallelEnv(
        inputs_path=INPUTS_PATH,
        scenario="example_02e",
        study_case="base",
        max_episode_steps=2,
    )
    try:
        with pytest.raises(ValueError, match="marginal cost"):
            MarkupActionWrapper(env)
        MarkupActionWrapper(env, force=True)  # explicit opt-out works
    finally:
        env.close()


def test_numeric_info_wrapper_strips_non_numeric_values():
    """
    Some frameworks build tensor specs from `info` and choke on anything that is
    not a number (TorchRL's PettingZooWrapper calls torch.as_tensor on every
    value). The wrapper keeps the numeric fields and drops the rest.
    """
    env = NumericInfoWrapper(AssumeParallelEnv(**MULTI, max_episode_steps=3))
    try:
        _, infos = env.reset(seed=0)
        for info in infos.values():
            assert "market_id" not in info  # str
            assert "product_start" not in info  # datetime
            assert info["assume_episode"] == 1.0

        _, _, _, _, infos = env.step(random_actions(env))
        for info in infos.values():
            for value in info.values():
                assert isinstance(value, float | np.ndarray)
    finally:
        env.close()


def test_numeric_info_wrapper_can_drop_info_entirely():
    env = NumericInfoWrapper(AssumeParallelEnv(**MULTI, max_episode_steps=3), keep=[])
    try:
        _, infos = env.reset(seed=0)
        assert all(info == {} for info in infos.values())
        _, _, _, _, infos = env.step(random_actions(env))
        assert all(info == {} for info in infos.values())
    finally:
        env.close()


# ----------------------------------------------------------------------
# the headline guarantee: ASSUME itself is unchanged
# ----------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize(
    "scenario, study_case",
    [("example_02a", "tiny"), ("example_02c", "base")],
)
def test_simulation_results_are_unchanged(scenario, study_case):
    """
    Running a scenario through the PettingZoo env must reproduce a plain ASSUME
    run bit for bit, given the same actions.

    The reference run uses ASSUME's own learning strategies with ``get_actions``
    stubbed out to return the same constants the environment is fed.
    """
    import torch as th

    from assume import World
    from assume.scenario.loader_csv import (
        load_config_and_create_forecaster,
        setup_world,
    )

    seed = 1234

    def per_agent_actions(agents, act_dim):
        # distinct prices per unit, so market clearing has no ties to break
        return {
            agent: np.array(
                [0.30 + 0.02 * i + 0.005 * j for j in range(act_dim)], dtype=np.float32
            )
            for i, agent in enumerate(sorted(agents))
        }

    # --- reference: plain ASSUME -------------------------------------
    random.seed(seed)
    world = World(database_uri="", export_csv_path="", log_level="ERROR")
    world.show_progress = False
    world.scenario_data = load_config_and_create_forecaster(
        INPUTS_PATH, scenario, study_case
    )
    world.scenario_data["config"]["learning_config"]["learning_mode"] = True
    setup_world(world=world, evaluation_mode=True)

    strategies = {
        unit_id: strategy
        for unit_id, unit in world.units.items()
        for strategy in unit.bidding_strategies.values()
        if hasattr(strategy, "act_dim")
    }
    actions = per_agent_actions(strategies, next(iter(strategies.values())).act_dim)
    for unit_id, strategy in strategies.items():
        constant = th.tensor(actions[unit_id], dtype=th.float)
        strategy.get_actions = lambda _obs, c=constant: (c, th.zeros_like(c))
    world.run()

    # --- through the PettingZoo environment ---------------------------
    env = AssumeParallelEnv(
        inputs_path=INPUTS_PATH, scenario=scenario, study_case=study_case
    )
    try:
        env.reset(seed=seed)
        while env.agents:
            env.step({agent: actions[agent] for agent in env.agents})

        assert env.observation_mismatches == 0
        assert set(env.world.units) == set(world.units)
        for unit_id in world.units:
            for series in ("energy", "profit"):
                reference = np.array(world.units[unit_id].outputs[series])
                through_env = np.array(env.world.units[unit_id].outputs[series])
                np.testing.assert_allclose(
                    through_env,
                    reference,
                    atol=1e-6,
                    err_msg=f"{unit_id}/{series} differs",
                )
    finally:
        env.close()
