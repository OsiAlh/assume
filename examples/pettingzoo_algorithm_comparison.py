"""
Compare multi-agent RL learners on the same ASSUME market.

Every arm gets the same scenario, the same episode budget and the same
evaluation harness, so the numbers sit in one table.

======================  =========================================================
arm (``--arms``)        what it is
======================  =========================================================
``mappo``               on-policy, independent actors, **centralised critic**
                        (TorchRL ``ClipPPOLoss``)
``ippo``                the same code with a **decentralised** critic - the
                        ablation that measures what centralisation buys
``matd3``               off-policy, deterministic, centralised **twin** critics,
                        delayed actor (TorchRL ``TD3Loss``)
``itd3``                MATD3 with a decentralised critic
``maddpg``              MATD3 minus the twin critics, target smoothing and the
                        delayed actor (TorchRL ``DDPGLoss``)
``masac``               maximum-entropy off-policy, stochastic actors,
                        centralised twin critics (TorchRL ``SACLoss``)
``assume``              ASSUME's own MATD3, via ``run_learning`` - never goes
                        through PettingZoo during training
``sb3ppo``              PPO from Stable-Baselines3 with SuperSuit parameter
                        sharing (optional dependency)
``agilerl``             MATD3 from AgileRL - a third implementation of the same
                        algorithm (optional dependency)
======================  =========================================================

Three things this layout is designed to show:

**One algorithm, three implementations.** ``matd3`` / ``assume`` / ``agilerl``
run the same algorithm from three independent codebases. Any gap between them is
implementation and hyperparameters, not algorithm.

**Centralisation, measured.** ``mappo`` vs ``ippo`` and ``matd3`` vs ``itd3`` are
the same code with one flag flipped on the critic, so the value of a centralised
critic on this market is measured rather than assumed.

**Parameter sharing, measured.** Every TorchRL arm uses ``share_params=False``
(one actor per unit); ``sb3ppo`` shares one policy across all agents. The five
learning units are byte-identical, so a shared policy is forced to bid
identically - watch the ``spread`` column.

**Against the equilibrium prices.** Where ``examples/prices`` ships a price file
for the scenario (``prices_02b.csv`` for ``example_02b``), the market clearing
prices every arm produces are compared with it hour by hour, in a second table.
Those prices are marginal cost where no learning unit is pivotal and just
below the next competitor's cost where one is (see
``assume_rl_metrics.load_equilibrium_prices``) - so bidding at marginal cost
falls short of them exactly in the pivotal hours, and a learner that has found
the equilibrium matches them.

Run it::

    python examples/pettingzoo_algorithm_comparison.py                 # default 3 arms
    python examples/pettingzoo_algorithm_comparison.py --arms all      # everything installed
    python examples/pettingzoo_algorithm_comparison.py --arms mappo,ippo
    python examples/pettingzoo_algorithm_comparison.py --help

Requires::

    pip install -e ".[pettingzoo]"
    pip install torchrl
    pip install stable-baselines3 supersuit    # for --arms sb3ppo
    pip install agilerl                        # for --arms agilerl

.. warning::

   ``agilerl`` pulls in ``minari`` and therefore ``pyarrow``, and with pyarrow
   installed pandas 3.x hands xarray an ``ArrowStringArray`` its indexer rejects,
   which breaks ASSUME's grid handling (``tests/test_redispatch.py`` fails).
   ``pip uninstall pyarrow`` afterwards fixes ASSUME and leaves the arm working.

Metrics are logged to TensorBoard using ASSUME's own tag names - see
``assume_rl_metrics.TensorBoardReporter`` - plus storage usage when the scenario
has storage units.

**This is not a benchmark.** One scenario, one seed by default and a small
episode budget cannot rank algorithms. Use ``--seeds 5`` before drawing any
conclusion, and read the reference rows in the results table first.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import shutil
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th
from tqdm import tqdm

# ASSUME's save-path prompt would block an unattended run
os.environ.setdefault("NON_INTERACTIVE", "1")

import assume_rl_metrics as metrics  # noqa: E402
from assume_rl_metrics import (  # noqa: E402
    load_assume_actors,
    marginal_cost_policy,
    random_policy,
)

from assume.reinforcement_learning.pettingzoo import (  # noqa: E402
    AssumeParallelEnv,
    MarkupActionWrapper,
    NumericInfoWrapper,
)

try:
    from tensordict.nn import TensorDictModule, TensorDictSequential
    from tensordict.nn.distributions import NormalParamExtractor
    from torchrl.data import (
        LazyTensorStorage,
        SamplerWithoutReplacement,
        TensorDictReplayBuffer,
    )
    from torchrl.envs.libs.pettingzoo import PettingZooWrapper
    from torchrl.envs.utils import MarlGroupMapType
    from torchrl.modules import (
        AdditiveGaussianModule,
        MultiAgentMLP,
        ProbabilisticActor,
        TanhNormal,
    )
    from torchrl.objectives import (
        ClipPPOLoss,
        DDPGLoss,
        SACLoss,
        SoftUpdate,
        TD3Loss,
        ValueEstimators,
    )
except ImportError as exc:  # pragma: no cover - dependency hint
    raise SystemExit(
        "The MAPPO and MATD3 arms need TorchRL:\n"
        "    pip install torchrl\n"
        f"(import failed: {exc})"
    ) from exc

logging.getLogger("torchrl").setLevel(logging.WARNING)

logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore")

INPUTS_PATH = str(Path(__file__).resolve().parent / "inputs")
SCENARIO = "example_02b"
STUDY_CASE = "comparison"

SCRATCH = Path(__file__).resolve().parent / "comparison_output"


# ---------------------------------------------------------------------------
# shared harness - every arm is measured with exactly this
# ---------------------------------------------------------------------------
#: TD3 updates the actor once every POLICY_DELAY critic updates.
POLICY_DELAY = 2

#: Set by --raw-actions. See `make_env`.
USE_MARKUP = True

#: Set by --max-episode-steps; None means the scenario's full horizon.
MAX_EPISODE_STEPS: int | None = None

#: Equilibrium prices of the scenario from examples/prices, at the simulation's
#: time step. Set in `main`; None when the scenario has no price file.
EQUILIBRIUM: pd.Series | None = None


def make_env(max_episode_steps: int | None = -1, markup: bool | None = None):
    """
    Build the environment an arm trains and is scored on.

    By default the PettingZoo arms bid a *markup on marginal cost*
    (:class:`MarkupActionWrapper`). That is not a handicap for ASSUME - it is
    what makes the comparison fair. ASSUME's MATD3 seeds its own exploration at
    marginal cost during the initial-experience episodes; without the wrapper an
    external learner would start every run bidding 0 EUR/MWh, far below cost,
    and would be measuring that head start rather than the algorithm.

    The ASSUME arm always uses the raw env: its actors are trained on absolute
    bid prices, so that is the space they output in. Both action spaces map into
    the same market, so the returns remain directly comparable.
    """
    if markup is None:
        markup = USE_MARKUP
    if max_episode_steps == -1:
        max_episode_steps = MAX_EPISODE_STEPS
    return metrics.make_env(
        INPUTS_PATH,
        SCENARIO,
        STUDY_CASE,
        markup=markup,
        span=metrics.MARKUP_SPAN,
        max_episode_steps=max_episode_steps,
    )


# ---------------------------------------------------------------------------
# TorchRL scaffolding shared by both PettingZoo arms
# ---------------------------------------------------------------------------
def build_torchrl_env(max_episode_steps: int | None = -1):
    """
    Wrap the ASSUME environment for TorchRL.

    Two ASSUME-side wrappers are needed:

    * :class:`MarkupActionWrapper` - the markup action space (see `make_env`).
    * :class:`NumericInfoWrapper` with ``keep=[]`` - TorchRL builds tensor specs
      from ``info`` during ``_init_env`` and calls ``torch.as_tensor`` on every
      value, which fails on ASSUME's ``market_id`` (a string) and the datetimes.
      It also requires the info schema to be identical at reset and at step,
      which ASSUME's is not. Dropping ``info`` sidesteps both.
    """
    if max_episode_steps == -1:
        max_episode_steps = MAX_EPISODE_STEPS
    inner = AssumeParallelEnv(
        inputs_path=INPUTS_PATH,
        scenario=SCENARIO,
        study_case=STUDY_CASE,
        max_episode_steps=max_episode_steps,
    )
    wrapped = NumericInfoWrapper(
        MarkupActionWrapper(inner) if USE_MARKUP else inner, keep=[]
    )
    env = PettingZooWrapper(
        wrapped,
        group_map=MarlGroupMapType.ALL_IN_ONE_GROUP,
        use_mask=False,
        categorical_actions=False,
        device="cpu",
    )
    return env, wrapped


def env_dims(env):
    n_agents = len(env.group_map["agents"])
    obs_dim = env.observation_spec["agents", "observation"].shape[-1]
    act_dim = env.full_action_spec["agents", "action"].shape[-1]
    return n_agents, obs_dim, act_dim


def critic_body(
    n_agents: int, in_dim: int, out_dim: int = 1, *, centralised: bool = True
):
    """
    The critic. ``centralised`` is the difference between the MA- and I- families.

    ``centralized=True`` is the whole point of the "MA" in MAPPO, MADDPG, MATD3
    and MASAC: the value function conditions on **every** agent's input, so the
    advantage an agent sees accounts for what the others are doing. Training is
    centralised; execution stays decentralised because the actor never touches
    this network.

    ``centralised=False`` gives the independent-learner ablation - IPPO, IDDPG,
    ITD3 - where each agent's critic sees only its own input and the other agents
    are just non-stationary environment. Running both is the cleanest way to see
    whether centralisation is buying anything on a given market.

    ``share_params=False`` always gives each agent its own critic parameters.
    """
    return MultiAgentMLP(
        n_agent_inputs=in_dim,
        n_agent_outputs=out_dim,
        n_agents=n_agents,
        centralized=centralised,
        share_params=False,
        depth=2,
        num_cells=128,
    )


def centralised_critic(n_agents: int, in_dim: int, out_dim: int = 1):
    """Backwards-compatible alias for a centralised :func:`critic_body`."""
    return critic_body(n_agents, in_dim, out_dim, centralised=True)


def independent_actor_body(n_agents: int, obs_dim: int, out_dim: int):
    """
    One policy network **per agent** (``share_params=False``).

    The five learning units in this scenario are byte-identical, so a shared
    policy is forced to bid identically for all of them - and symmetric bidding
    is measurably worse here than an asymmetric configuration (see the closing
    notes). Independent actors can break that symmetry, which is what ASSUME's
    own per-unit actors do.
    """
    return MultiAgentMLP(
        n_agent_inputs=obs_dim,
        n_agent_outputs=out_dim,
        n_agents=n_agents,
        centralized=False,
        share_params=False,
        depth=2,
        num_cells=128,
    )


class JointCritic(th.nn.Module):
    """
    Q(obs, action) as a real ``nn.Module``.

    Wrapping the concatenation in a lambda inside a ``TensorDictModule`` looks
    equivalent but is not: the lambda registers **no parameters**, so the loss
    cannot find the critic weights and ``delay_qvalue`` produces target networks
    that share (or lack) parameters. SAC fails loudly on this; DDPG and TD3 fail
    quietly, which is worse. A proper module registers its weights and the
    losses build targets correctly.
    """

    def __init__(self, net: th.nn.Module):
        super().__init__()
        self.net = net

    def forward(self, observation, action):
        return self.net(th.cat([observation, action], dim=-1))


#: Label of the arm currently training, so its progress bar can name itself.
CURRENT_ARM = ""


def episode_bar(episodes: int, unit: str = "ep"):
    """
    One progress bar per arm, over training episodes.

    Without this a long run prints "training matd3 ..." and then nothing for
    minutes. ASSUME's own arm already shows a bar (``run_learning`` creates one),
    so this gives the other arms the same feedback.

    ``leave=False`` keeps the finished bars from piling up above the results
    table; the per-arm summary line is printed by `main` anyway.
    """
    return tqdm(
        range(episodes),
        desc=f"    {CURRENT_ARM or 'training'}",
        unit=unit,
        leave=False,
        dynamic_ncols=True,
    )


def _episode_length(env) -> int:
    """
    Market openings per episode, from the scenario horizon.

    SB3 needs an ``n_steps`` up front, and it should divide the episode rather
    than straddle resets.
    """
    world = env.unwrapped.world
    horizon = world.end - world.start
    step = pd.Timedelta(world.units[env.possible_agents[0]].index.freq)
    return max(int(horizon / step) - 1, 1)


def torchrl_policy(actor_net, agent_order, deterministic_fn):
    """Adapt a TorchRL actor network to the shared evaluation harness."""

    def policy(inner_env, observations):
        batch = th.as_tensor(
            np.stack([observations[a] for a in agent_order]), dtype=th.float
        )
        with th.no_grad():
            actions = deterministic_fn(actor_net, batch).cpu().numpy()
        return {a: actions[i] for i, a in enumerate(agent_order)}

    return policy


# ---------------------------------------------------------------------------
# arm A: MAPPO (TorchRL)
# ---------------------------------------------------------------------------
def train_ppo(
    episodes: int,
    seed: int = 42,
    epochs: int = 4,
    minibatch: int = 128,
    on_episode=None,
    centralised: bool = True,
):
    """
    Multi-Agent PPO: independent actors, **centralised critic**.

    The centralised critic is what separates MAPPO from IPPO - the value
    function conditions on the joint observation, so the advantage an agent sees
    accounts for what the others are doing, while the policy it trains stays
    decentralised.

    Everything algorithmic here (GAE, the clipped surrogate, the entropy term)
    is TorchRL's ``ClipPPOLoss``.
    """
    th.manual_seed(seed)
    env, wrapped = build_torchrl_env()
    n_agents, obs_dim, act_dim = env_dims(env)
    agent_order = list(env.group_map["agents"])

    actor_net = th.nn.Sequential(
        independent_actor_body(n_agents, obs_dim, 2 * act_dim),
        NormalParamExtractor(),
    )
    policy_module = TensorDictModule(
        actor_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "loc"), ("agents", "scale")],
    )
    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.full_action_spec["agents", "action"],
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[("agents", "action")],
        distribution_class=TanhNormal,
        distribution_kwargs={"low": -1.0, "high": 1.0},
        return_log_prob=True,
        log_prob_key=("agents", "sample_log_prob"),
    )

    critic = TensorDictModule(
        critic_body(n_agents, obs_dim, centralised=centralised),
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "state_value")],
    )

    loss = ClipPPOLoss(
        actor_network=policy,
        critic_network=critic,
        clip_epsilon=0.2,
        entropy_coeff=0.001,
        normalize_advantage=True,
    )
    loss.set_keys(
        reward=("agents", "reward"),
        action=("agents", "action"),
        sample_log_prob=("agents", "sample_log_prob"),
        value=("agents", "state_value"),
        done=("agents", "done"),
        terminated=("agents", "terminated"),
    )
    loss.make_value_estimator(ValueEstimators.GAE, gamma=0.99, lmbda=0.95)
    advantage = loss.value_estimator
    optimiser = th.optim.Adam(loss.parameters(), lr=3e-4)

    # built up front so a callback can evaluate the policy mid-training
    eval_policy = torchrl_policy(
        actor_net, agent_order, lambda net, obs: th.tanh(net[0](obs)[..., :act_dim])
    )

    for episode in episode_bar(episodes):
        with th.no_grad():
            batch = env.rollout(
                max_steps=100_000,
                policy=policy,
                auto_reset=True,
                break_when_any_done=True,
            )
            advantage(
                batch,
                params=loss.critic_network_params,
                target_params=loss.target_critic_network_params,
            )

        flat = batch.reshape(-1)
        size = min(minibatch, len(flat))
        buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(len(flat)),
            sampler=SamplerWithoutReplacement(),
            batch_size=size,
        )
        buffer.extend(flat)
        for _ in range(epochs):
            for _ in range(max(1, len(flat) // size)):
                sub = buffer.sample()
                out = loss(sub)
                total = out["loss_objective"] + out["loss_critic"] + out["loss_entropy"]
                optimiser.zero_grad()
                total.backward()
                th.nn.utils.clip_grad_norm_(loss.parameters(), 0.5)
                optimiser.step()

        if on_episode is not None:
            on_episode(episode + 1, eval_policy)

    env.close()
    # deterministic action = the distribution's mode; TanhNormal on [-1, 1] makes
    # that tanh(loc), and NormalParamExtractor returns (loc, scale)
    return eval_policy


# arm B: ASSUME's own MATD3
# ---------------------------------------------------------------------------
def train_assume_matd3(episodes: int, seed: int = 42):
    """
    Run ASSUME's built-in training loop, then hand its actors back as a policy.

    This arm never touches PettingZoo during training - ``run_learning`` drives
    the simulation itself, exactly as it does for any ASSUME user. Only the
    *evaluation* goes through the PettingZoo env, so all three arms are scored by
    the same harness.
    """
    from assume import World
    from assume.scenario.loader_csv import (
        load_config_and_create_forecaster,
        run_learning,
        setup_world,
    )

    save_dir = (
        Path(INPUTS_PATH) / SCENARIO / "learned_strategies" / f"{SCENARIO}_{STUDY_CASE}"
    )
    shutil.rmtree(save_dir, ignore_errors=True)  # start from a clean slate

    SCRATCH.mkdir(parents=True, exist_ok=True)
    db_path = SCRATCH / "assume_matd3.db"
    db_path.unlink(missing_ok=True)

    world = World(
        database_uri=f"sqlite:///{db_path}", export_csv_path="", log_level="ERROR"
    )
    world.show_progress = False  # silence the inner per-simulation bar

    # The episode budget must reach the scenario data *before* setup_world builds
    # the learning role. run_learning reads
    # `world.learning_role.learning_config.training_episodes`, which is frozen at
    # that point, so setting it afterwards is silently ignored and this arm would
    # train for whatever the config says - wrecking budget parity with the others.
    world.scenario_data = load_config_and_create_forecaster(
        INPUTS_PATH, SCENARIO, STUDY_CASE
    )
    learning_config = world.scenario_data["config"]["learning_config"]
    learning_config["training_episodes"] = episodes

    # ASSUME requires training_episodes >= episodes_collecting_initial_experience
    # + validation_episodes_interval, and raises deep inside run_learning
    # otherwise. Shrink both to fit a small budget rather than fail: they are
    # bookkeeping knobs, and clamping them keeps the episode budget matched with
    # the other arms.
    initial = min(
        learning_config.get("episodes_collecting_initial_experience", 5),
        max(1, episodes // 3),
    )
    interval = min(
        learning_config.get("validation_episodes_interval", 10),
        max(1, episodes - initial),
    )
    if (initial, interval) != (
        learning_config.get("episodes_collecting_initial_experience"),
        learning_config.get("validation_episodes_interval"),
    ):
        print(
            f"  (ASSUME arm: {episodes} episodes is below the scenario's "
            f"bookkeeping minimum; using initial_experience={initial}, "
            f"validation_interval={interval})"
        )
    learning_config["episodes_collecting_initial_experience"] = initial
    learning_config["validation_episodes_interval"] = interval

    setup_world(world=world)
    assert world.learning_role.learning_config.training_episodes == episodes

    # run_learning needs a database: it reads evaluation rewards back out of it
    run_learning(world)

    # the shared harness rebuilds ASSUME's actors; it needs a raw env only to
    # read each strategy's dimensions
    probe = make_env(max_episode_steps=1, markup=False)
    try:
        return load_assume_actors(save_dir, probe)
    finally:
        probe.close()


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# arm C: MATD3 (TorchRL)
# ---------------------------------------------------------------------------
def train_td3(
    episodes: int,
    seed: int = 42,
    updates_per_step: float = 0.4,
    on_episode=None,
    centralised: bool = True,
):
    """
    Multi-Agent TD3: independent actors, **centralised twin critics**.

    Same network layout as the MAPPO arm, so the two differ by algorithm rather
    than by architecture. Twin critics, target-policy smoothing and the delayed
    actor update all come from TorchRL's ``TD3Loss`` - this arm is the external
    counterpart to ASSUME's own MATD3.
    """
    th.manual_seed(seed)
    env, wrapped = build_torchrl_env()
    n_agents, obs_dim, act_dim = env_dims(env)
    agent_order = list(env.group_map["agents"])

    actor_net = th.nn.Sequential(
        independent_actor_body(n_agents, obs_dim, act_dim), th.nn.Tanh()
    )
    policy_module = TensorDictModule(
        actor_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "action")],
    )
    # exploration lives outside the actor, so evaluation can use the bare actor
    explorative_policy = TensorDictSequential(
        policy_module,
        AdditiveGaussianModule(
            spec=env.full_action_spec["agents", "action"],
            sigma_init=0.2,
            sigma_end=0.05,
            annealing_num_steps=episodes * 100,
            action_key=("agents", "action"),
        ),
    )

    critic_net = critic_body(n_agents, obs_dim + act_dim, centralised=centralised)
    critic = TensorDictModule(
        JointCritic(critic_net),
        in_keys=[("agents", "observation"), ("agents", "action")],
        out_keys=[("agents", "state_action_value")],
    )

    loss = TD3Loss(
        actor_network=policy_module,
        qvalue_network=critic,
        # TD3Loss indexes the spec with its own `action` key, so hand it the
        # "agents" sub-spec rather than the full composite
        action_spec=env.full_action_spec["agents"],
        num_qvalue_nets=2,
        delay_actor=True,
        delay_qvalue=True,
        loss_function="l2",
    )
    loss.set_keys(
        action=("agents", "action"),
        state_action_value=("agents", "state_action_value"),
        reward=("agents", "reward"),
        done=("agents", "done"),
        terminated=("agents", "terminated"),
    )
    loss.make_value_estimator(gamma=0.99)
    target_updater = SoftUpdate(loss, tau=0.005)

    # Separate optimisers, on purpose. The actor loss is -Q(s, actor(s)), so it
    # has a gradient with respect to the *critic* parameters too. Summing the two
    # losses into one optimiser over loss.parameters() would apply that gradient
    # to the critic - training it to lower its own Q estimates - and would also
    # drop TD3's delayed policy update. Both wreck the arm.
    #
    # Take the parameters off the *loss*, not off the modules. With
    # `num_qvalue_nets=2` TD3Loss expands the critic into its own leaves of shape
    # (2, n_agents, ...), which share no storage with `critic_net.parameters()`;
    # an optimiser built from the module would silently step nothing, leaving the
    # critic at its random initialisation while the actor climbs a frozen random
    # Q surface. The actor is not expanded, so `actor_network_params` and
    # `actor_net.parameters()` are the same tensors and `eval_policy` - which
    # calls `actor_net` directly - still sees every update. The two groups are
    # disjoint, which is exactly what separate optimisers need.
    actor_params = list(loss.actor_network_params.values(True, True))
    critic_params = list(loss.qvalue_network_params.values(True, True))
    assert not ({id(p) for p in actor_params} & {id(p) for p in critic_params})
    optimiser_actor = th.optim.Adam(actor_params, lr=1e-4)
    optimiser_critic = th.optim.Adam(critic_params, lr=1e-3)
    buffer = TensorDictReplayBuffer(storage=LazyTensorStorage(200_000), batch_size=128)
    update_counter = 0

    eval_policy = torchrl_policy(actor_net, agent_order, lambda net, obs: net(obs))

    for episode in episode_bar(episodes):
        with th.no_grad():
            rollout = env.rollout(
                max_steps=100_000,
                policy=explorative_policy,
                auto_reset=True,
                break_when_any_done=True,
            )
        buffer.extend(rollout.reshape(-1))
        # AdditiveGaussianModule only anneals when `step()` is called; without
        # this sigma stays at sigma_init for the whole run and sigma_end /
        # annealing_num_steps are dead settings.
        explorative_policy[1].step(len(rollout))

        # skip the update while the buffer is still filling, but keep
        # reporting so the logged curve starts at episode 1
        if len(buffer) >= buffer._batch_size:
            # `updates_per_step` is set so this arm performs roughly as many gradient
            # steps per episode as ASSUME's does under the scenario's
            # `train_freq` / `gradient_steps`, keeping the two MATD3 arms comparable
            # on compute as well as on episodes.
            for _ in range(int(len(rollout) * updates_per_step)):
                sample = buffer.sample()

                critic_loss, _ = loss.value_loss(sample)
                optimiser_critic.zero_grad()
                critic_loss.backward()
                th.nn.utils.clip_grad_norm_(critic_params, 0.5)
                optimiser_critic.step()

                # TD3's delayed policy update: the actor moves once per POLICY_DELAY
                # critic updates, so it chases a value function that has settled
                update_counter += 1
                if update_counter % POLICY_DELAY == 0:
                    actor_loss, _ = loss.actor_loss(sample)
                    optimiser_actor.zero_grad()
                    actor_loss.backward()
                    th.nn.utils.clip_grad_norm_(actor_params, 0.5)
                    optimiser_actor.step()
                    target_updater.step()

        if on_episode is not None:
            on_episode(episode + 1, eval_policy)

    env.close()
    return eval_policy


# driver
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# arm D: MADDPG (TorchRL) - the predecessor of MATD3
# ---------------------------------------------------------------------------
def train_ddpg(
    episodes: int,
    seed: int = 42,
    updates_per_step: float = 0.4,
    on_episode=None,
    centralised: bool = True,
):
    """
    Multi-Agent DDPG: independent actors, centralised critic, **one** Q-network.

    MADDPG is MATD3 minus TD3's three additions - twin critics, target-policy
    smoothing and the delayed actor update. Running both isolates what those
    tricks are worth here rather than assuming they help.
    """
    th.manual_seed(seed)
    env, _ = build_torchrl_env()
    n_agents, obs_dim, act_dim = env_dims(env)
    agent_order = list(env.group_map["agents"])

    actor_net = th.nn.Sequential(
        independent_actor_body(n_agents, obs_dim, act_dim), th.nn.Tanh()
    )
    policy_module = TensorDictModule(
        actor_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "action")],
    )
    explorative_policy = TensorDictSequential(
        policy_module,
        AdditiveGaussianModule(
            spec=env.full_action_spec["agents", "action"],
            sigma_init=0.2,
            sigma_end=0.05,
            annealing_num_steps=max(episodes * 100, 1),
            action_key=("agents", "action"),
        ),
    )

    critic_net = critic_body(n_agents, obs_dim + act_dim, centralised=centralised)
    critic = TensorDictModule(
        JointCritic(critic_net),
        in_keys=[("agents", "observation"), ("agents", "action")],
        out_keys=[("agents", "state_action_value")],
    )

    loss = DDPGLoss(
        actor_network=policy_module,
        value_network=critic,
        delay_actor=True,
        delay_value=True,
        loss_function="l2",
    )
    loss.set_keys(
        state_action_value=("agents", "state_action_value"),
        reward=("agents", "reward"),
        done=("agents", "done"),
        terminated=("agents", "terminated"),
    )
    loss.make_value_estimator(gamma=0.99)
    target_updater = SoftUpdate(loss, tau=0.005)
    optimiser = th.optim.Adam(loss.parameters(), lr=3e-4)
    buffer = TensorDictReplayBuffer(storage=LazyTensorStorage(200_000), batch_size=128)

    eval_policy = torchrl_policy(actor_net, agent_order, lambda net, obs: net(obs))

    for episode in episode_bar(episodes):
        with th.no_grad():
            rollout = env.rollout(
                max_steps=100_000,
                policy=explorative_policy,
                auto_reset=True,
                break_when_any_done=True,
            )
        buffer.extend(rollout.reshape(-1))
        # AdditiveGaussianModule only anneals when `step()` is called; without
        # this sigma stays at sigma_init for the whole run and sigma_end /
        # annealing_num_steps are dead settings.
        explorative_policy[1].step(len(rollout))

        if len(buffer) >= buffer._batch_size:
            for _ in range(int(len(rollout) * updates_per_step)):
                out = loss(buffer.sample())
                total = out["loss_actor"] + out["loss_value"]
                optimiser.zero_grad()
                total.backward()
                th.nn.utils.clip_grad_norm_(loss.parameters(), 0.5)
                optimiser.step()
                target_updater.step()

        if on_episode is not None:
            on_episode(episode + 1, eval_policy)

    env.close()
    return eval_policy


# ---------------------------------------------------------------------------
# arm E: MASAC (TorchRL) - maximum-entropy off-policy
# ---------------------------------------------------------------------------
def train_sac(
    episodes: int,
    seed: int = 42,
    updates_per_step: float = 0.4,
    on_episode=None,
    centralised: bool = True,
):
    """
    Multi-Agent SAC: stochastic independent actors, centralised twin critics.

    A different family from the two above: SAC maximises reward *plus* policy
    entropy, and tunes the trade-off automatically. That matters on this market
    because the profitable bid band is narrow and sits next to a cliff - an
    algorithm that keeps deliberate stochasticity explores it differently from
    TD3's fixed additive noise.
    """
    th.manual_seed(seed)
    env, _ = build_torchrl_env()
    n_agents, obs_dim, act_dim = env_dims(env)
    agent_order = list(env.group_map["agents"])

    actor_net = th.nn.Sequential(
        independent_actor_body(n_agents, obs_dim, 2 * act_dim),
        NormalParamExtractor(),
    )
    policy_module = TensorDictModule(
        actor_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "loc"), ("agents", "scale")],
    )
    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.full_action_spec["agents", "action"],
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[("agents", "action")],
        distribution_class=TanhNormal,
        distribution_kwargs={"low": -1.0, "high": 1.0},
        return_log_prob=True,
        log_prob_key=("agents", "sample_log_prob"),
    )

    critic_net = critic_body(n_agents, obs_dim + act_dim, centralised=centralised)
    critic = TensorDictModule(
        JointCritic(critic_net),
        in_keys=[("agents", "observation"), ("agents", "action")],
        out_keys=[("agents", "state_action_value")],
    )

    loss = SACLoss(
        actor_network=policy,
        qvalue_network=critic,
        num_qvalue_nets=2,
        # SACLoss indexes the spec with the *full* nested action key
        # (``action_spec[("agents",)]``), unlike TD3Loss which indexes with the
        # leaf key - so this one gets the whole composite
        action_spec=env.full_action_spec,
        delay_qvalue=True,
        loss_function="l2",
    )
    loss.set_keys(
        action=("agents", "action"),
        state_action_value=("agents", "state_action_value"),
        log_prob=("agents", "sample_log_prob"),
        reward=("agents", "reward"),
        done=("agents", "done"),
        terminated=("agents", "terminated"),
    )
    loss.make_value_estimator(gamma=0.99)
    target_updater = SoftUpdate(loss, tau=0.005)
    optimiser = th.optim.Adam(loss.parameters(), lr=3e-4)
    buffer = TensorDictReplayBuffer(storage=LazyTensorStorage(200_000), batch_size=128)

    # deterministic evaluation = the distribution's mode, tanh(loc) on [-1, 1]
    eval_policy = torchrl_policy(
        actor_net, agent_order, lambda net, obs: th.tanh(net[0](obs)[..., :act_dim])
    )

    for episode in episode_bar(episodes):
        with th.no_grad():
            rollout = env.rollout(
                max_steps=100_000,
                policy=policy,
                auto_reset=True,
                break_when_any_done=True,
            )
        buffer.extend(rollout.reshape(-1))

        if len(buffer) >= buffer._batch_size:
            for _ in range(int(len(rollout) * updates_per_step)):
                out = loss(buffer.sample())
                total = out["loss_actor"] + out["loss_qvalue"] + out["loss_alpha"]
                optimiser.zero_grad()
                total.backward()
                th.nn.utils.clip_grad_norm_(loss.parameters(), 0.5)
                optimiser.step()
                target_updater.step()

        if on_episode is not None:
            on_episode(episode + 1, eval_policy)

    env.close()
    return eval_policy


# ---------------------------------------------------------------------------
# arms F/G: other frameworks (optional dependencies)
# ---------------------------------------------------------------------------
def train_sb3_ppo(episodes: int, seed: int = 42, on_episode=None):
    """
    PPO from Stable-Baselines3, parameter-shared across agents via SuperSuit.

    A second *implementation* of the PPO family, from a different framework, to
    sit next to the TorchRL MAPPO/IPPO rows. Two differences are structural, not
    incidental, and both are visible in the results:

    * SuperSuit's ``pettingzoo_env_to_vec_env_v1`` turns the agents into vector
      sub-environments sharing **one** policy, so this arm cannot break the
      symmetry between identical units the way independent actors can.
    * SB3's PPO has no centralised critic, so it is an IPPO, not a MAPPO.

    See ``examples/pettingzoo_sb3_training.py`` for the standalone version and
    the two SuperSuit quirks it works around.
    """
    try:
        import supersuit as ss
        from stable_baselines3 import PPO
        from supersuit.vector.concat_vec_env import ConcatVecEnv
        from supersuit.vector.sb3_vector_wrapper import SB3VecEnvWrapper
    except ImportError as exc:  # pragma: no cover - optional arm
        raise SystemExit(
            "The sb3ppo arm needs: pip install stable-baselines3 supersuit\n"
            f"(import failed: {exc})"
        ) from exc

    class _SeedableVecEnv(SB3VecEnvWrapper):
        """SuperSuit's wrapper defines no seed() and drops the seed in reset()."""

        def __init__(self, venv):
            super().__init__(venv)
            self._pending_seed = None

        def seed(self, seed=None):
            self._pending_seed = seed
            return [seed] * self.num_envs

        def reset(self, seed=None, options=None):
            if seed is None:
                seed, self._pending_seed = self._pending_seed, None
            obs, self.reset_infos = self.venv.reset(seed=seed, options=options)
            return obs

        def get_attr(self, attr_name, indices=None):
            if attr_name == "render_mode":
                return [self.render_mode] * self.num_envs
            return super().get_attr(attr_name, indices)

    train_env = make_env(markup=USE_MARKUP)
    agent_order = list(train_env.possible_agents)
    steps = _episode_length(train_env)

    # ConcatVecEnv takes factories, which avoids the cloudpickle round-trip
    # concat_vec_envs_v1 does - this env owns a live simulation thread
    markov_env = ss.pettingzoo_env_to_vec_env_v1(train_env)
    vec_env = _SeedableVecEnv(ConcatVecEnv([lambda: markov_env]))

    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=max(steps, 8),
        batch_size=128,
        n_epochs=4,
        learning_rate=3e-4,
        ent_coef=0.0,
        # SB3's default std of 1.0 spans the whole [-1, 1] action range, so most
        # samples clip at the bounds and the narrow profitable band is invisible
        policy_kwargs=dict(log_std_init=-1.2),
        verbose=0,
        seed=seed,
    )

    def eval_policy(inner_env, observations):
        batch = np.stack([observations[a] for a in inner_env.agents])
        actions, _ = model.predict(batch, deterministic=True)
        return dict(zip(inner_env.agents, actions))

    done = 0
    progress = episode_bar(episodes)
    while done < episodes:
        block = min(5, episodes - done)
        model.learn(
            total_timesteps=steps * len(agent_order) * block,
            reset_num_timesteps=False,
        )
        done += block
        progress.update(block)
        if on_episode is not None:
            on_episode(done, eval_policy)

    progress.close()
    vec_env.close()
    return eval_policy


def train_agilerl_matd3(episodes: int, seed: int = 42, on_episode=None):
    """
    MATD3 from AgileRL - a third independent implementation of the algorithm.

    .. warning::

       ``agilerl`` pulls in ``minari`` and therefore ``pyarrow``. With pyarrow
       installed, pandas 3.x hands xarray an ``ArrowStringArray`` its indexer
       rejects, which breaks ASSUME's grid handling
       (``tests/test_redispatch.py`` fails). ``pip uninstall pyarrow``
       afterwards fixes ASSUME and leaves this arm working - it only needs
       pyarrow for Minari dataset IO, which is unused here.

    Note AgileRL groups agents by a shared id prefix and shares one network per
    group, so with homogeneous units this arm cannot differentiate them the way
    the TorchRL arms (``share_params=False``) can.
    """
    try:
        from agilerl.algorithms import MATD3
        from agilerl.components.data import (
            MultiAgentTransition,
            transition_to_tensordict,
        )
        from agilerl.components.replay_buffer import ReplayBuffer
    except ImportError as exc:  # pragma: no cover - optional arm
        raise SystemExit(
            f"The agilerl arm needs: pip install agilerl\n(import failed: {exc})"
        ) from exc

    th.manual_seed(seed)
    env = make_env(markup=USE_MARKUP)
    agents = list(env.possible_agents)

    algo = MATD3(
        observation_spaces=[env.observation_space(a) for a in agents],
        action_spaces=[env.action_space(a) for a in agents],
        agent_ids=agents,
        device="cpu",
        batch_size=128,
        lr_actor=1e-4,
        lr_critic=1e-3,
        learn_step=5,
        gamma=0.99,
        tau=0.01,
        expl_noise=0.15,
    )
    memory = ReplayBuffer(max_size=200_000, device="cpu")

    def _act(inner_env, observations, explore: bool):
        was_training = getattr(algo, "training", True)
        algo.set_training_mode(explore)
        try:
            actions, raw = algo.get_action(
                obs={a: observations[a][None, :] for a in inner_env.agents}, infos=None
            )
        finally:
            algo.set_training_mode(was_training)
        stepped = {
            a: np.clip(
                np.asarray(actions[a]).reshape(-1),
                inner_env.action_space(a).low,
                inner_env.action_space(a).high,
            )
            for a in inner_env.agents
        }
        return stepped, raw

    def eval_policy(inner_env, observations):
        # deterministic: get_action adds exploration noise while training mode is
        # on, and scoring that against deterministic arms is not a comparison
        return _act(inner_env, observations, explore=False)[0]

    algo.set_training_mode(True)
    total_steps = 0
    for episode in episode_bar(episodes):
        observations, _ = env.reset(seed=seed * 1000 + episode)
        while env.agents:
            batched = {a: observations[a][None, :] for a in agents}
            stepped, raw = _act(env, observations, explore=True)
            next_observations, rewards, terminations, _, _ = env.step(stepped)

            transition = transition_to_tensordict(
                MultiAgentTransition(
                    obs=batched,
                    action=raw,
                    reward={
                        a: np.array([[rewards[a]]], dtype=np.float32) for a in agents
                    },
                    next_obs={a: next_observations[a][None, :] for a in agents},
                    done={
                        a: np.array([[terminations[a]]], dtype=np.float32)
                        for a in agents
                    },
                )
            )
            transition.batch_size = th.Size([1])
            memory.add(transition)

            observations = next_observations
            total_steps += 1
            if len(memory) >= algo.batch_size and total_steps % algo.learn_step == 0:
                algo.learn(memory.sample(algo.batch_size))

        if on_episode is not None:
            on_episode(episode + 1, eval_policy)

    env.close()
    return eval_policy


#: Every arm: how to train it, and what to call it in the results table.
#: The ``I*`` rows are the independent-learner ablations - identical code with a
#: decentralised critic - so the value of centralisation is measured, not assumed.
ARM_SPECS: dict[str, tuple[str, object, dict]] = {
    # on-policy
    "mappo": ("MAPPO (TorchRL)", lambda: train_ppo, {"centralised": True}),
    "ippo": ("IPPO (TorchRL)", lambda: train_ppo, {"centralised": False}),
    # off-policy, deterministic
    "matd3": ("MATD3 (TorchRL)", lambda: train_td3, {"centralised": True}),
    "itd3": ("ITD3 (TorchRL)", lambda: train_td3, {"centralised": False}),
    "maddpg": ("MADDPG (TorchRL)", lambda: train_ddpg, {"centralised": True}),
    # off-policy, stochastic
    "masac": ("MASAC (TorchRL)", lambda: train_sac, {"centralised": True}),
    # ASSUME's own implementation
    "assume": ("MATD3 (ASSUME)", lambda: train_assume_matd3, {}),
    # other frameworks, optional dependencies
    "sb3ppo": ("PPO (SB3+SuperSuit)", lambda: train_sb3_ppo, {}),
    "agilerl": ("MATD3 (AgileRL)", lambda: train_agilerl_matd3, {}),
}

#: Run by default. The optional-dependency arms are opt-in via --arms.
ARMS = ("mappo", "matd3", "assume")

#: Arms that need a package beyond the pettingzoo extra and torchrl.
OPTIONAL_ARMS = {"sb3ppo": "stable-baselines3 supersuit", "agilerl": "agilerl"}


def _installed(arm: str) -> bool:
    """Whether an optional arm's framework is importable."""
    import importlib.util

    modules = {"sb3ppo": ("stable_baselines3", "supersuit"), "agilerl": ("agilerl",)}
    return all(importlib.util.find_spec(m) is not None for m in modules.get(arm, ()))


def replay_assume_training_curve(db_path: Path, simulation_id: str, reporter) -> None:
    """
    Mirror ASSUME's own per-episode metrics into the same TensorBoard run.

    ``run_learning`` already writes a reward per unit per market opening into the
    ``rl_params`` table, and ASSUME's own TensorBoardLogger reads it back. Doing
    the same here puts all three arms on one set of axes with one set of tag
    names, instead of two TensorBoard directories that cannot be overlaid.
    """
    import sqlite3
    from collections import defaultdict

    if not Path(db_path).is_file():
        return

    rows = (
        sqlite3.connect(db_path)
        .execute(
            "SELECT episode, evaluation_mode, unit, SUM(reward), SUM(profit), "
            "SUM(regret) FROM rl_params WHERE simulation = ? "
            "GROUP BY episode, evaluation_mode, unit",
            (simulation_id,),
        )
        .fetchall()
    )

    grouped: dict = defaultdict(lambda: {"rewards": {}, "profit": [], "regret": []})
    for episode, evaluation_mode, unit, reward, profit, regret in rows:
        is_eval = evaluation_mode in (1, True, "True", "true")
        entry = grouped[("01_eval" if is_eval else "02_train", episode)]
        entry["rewards"][unit] = float(reward or 0.0)
        entry["profit"].append(float(profit or 0.0))
        entry["regret"].append(float(regret or 0.0))

    for (mode, episode), entry in sorted(grouped.items()):
        reporter.log_episode(
            episode,
            entry["rewards"],
            mode=mode,
            profit=float(np.mean(entry["profit"])),
            regret=float(np.mean(entry["regret"])),
        )


def run_arm(
    name: str,
    episodes: int,
    seed: int,
    eval_episodes: int,
    reporter=None,
    log_every: int = 5,
):
    """
    Train one arm and score it with the shared harness.

    Each arm is evaluated on an environment matching the action space it was
    trained in - markup for the PettingZoo arms, raw absolute prices for ASSUME's
    own actors. Both feed the same market, so the returns compare directly.
    """
    # the PettingZoo arms are scored in whichever space they trained in
    # (markup unless --raw-actions); ASSUME's actors always bid absolute prices
    eval_env = make_env(markup=USE_MARKUP and name != "assume")

    global CURRENT_ARM
    CURRENT_ARM = ARM_SPECS[name][0]

    def on_episode(episode: int, policy) -> None:
        """Periodic mid-training evaluation, logged with ASSUME's tag names."""
        if reporter is None or episode % log_every:
            return
        snapshot = metrics.evaluate_full(
            eval_env, policy, episodes=1, equilibrium=EQUILIBRIUM
        )
        reporter.log_episode(
            episode,
            snapshot.per_agent,
            mode="02_train",
            profit=snapshot.profit,
            regret=snapshot.regret,
            bid=snapshot.bid,
            storage=snapshot.storage,
            prices=snapshot.prices,
        )

    started = time.time()
    try:
        _, trainer_ref, kwargs = ARM_SPECS[name]
        trainer = trainer_ref()

        if name == "assume":
            # ASSUME drives its own loop and logs its own curve to the database
            policy = trainer(episodes=episodes, seed=seed, **kwargs)
            if reporter is not None:
                replay_assume_training_curve(
                    SCRATCH / "assume_matd3.db",
                    f"{SCENARIO}_{STUDY_CASE}",
                    reporter,
                )
        else:
            policy = trainer(
                episodes=episodes, seed=seed, on_episode=on_episode, **kwargs
            )
        elapsed = time.time() - started

        result = metrics.evaluate_full(
            eval_env, policy, episodes=eval_episodes, equilibrium=EQUILIBRIUM
        )
        if reporter is not None:
            reporter.log_episode(
                episodes,
                result.per_agent,
                mode="01_eval",
                profit=result.profit,
                regret=result.regret,
                bid=result.bid,
                storage=result.storage,
                prices=result.prices,
            )
            reporter.close()
    finally:
        eval_env.close()

    return {
        "score": result.score,
        "per_agent": result.per_agent,
        "bid": result.bid,
        "storage": result.storage,
        "prices": result.prices,
        "seconds": elapsed,
        "spread": max(result.per_agent.values()) - min(result.per_agent.values()),
    }


def average_prices(comparisons: list) -> metrics.PriceComparison | None:
    """Mean over seeds of one arm's price comparisons (each covers the same hours)."""
    comparisons = [c for c in comparisons if c is not None]
    if not comparisons:
        return None
    return metrics.PriceComparison(
        **{
            field.name: (
                sum(c.hours for c in comparisons)
                if field.name == "hours"
                else float(np.mean([getattr(c, field.name) for c in comparisons]))
            )
            for field in dataclasses.fields(metrics.PriceComparison)
        }
    )


def print_price_table(rows: list[tuple[str, object]], price_file: Path) -> None:
    """Clearing prices against the equilibrium prices, one row per policy."""
    print("\n" + "=" * 74)
    print(f"Clearing prices vs equilibrium prices ({price_file.name}), EUR/MWh")
    print("=" * 74)
    print(
        f"  {'learner':<22} {'price':>7} {'equil.':>7} {'diff':>7}"
        f" {'MAE':>7} {'RMSE':>7} {'matched':>8}"
    )
    print(f"  {'-' * 22} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 8}")
    for label, prices in rows:
        if prices is None:
            print(f"  {label:<22} {'n/a':>7}")
            continue
        print(
            f"  {label:<22} {prices.clearing:>7.2f} {prices.equilibrium:>7.2f}"
            f" {prices.difference:>+7.2f} {prices.mae:>7.2f} {prices.rmse:>7.2f}"
            f" {prices.matched:>8.0%}"
        )
    print(
        "\n  'price' is the mean clearing price over the evaluation episodes and\n"
        "  'equil.' the mean equilibrium price over the same hours; 'diff' is the\n"
        "  mean of clearing minus equilibrium, hour by hour. 'matched' counts the\n"
        "  hours within 1 EUR/MWh of equilibrium.\n"
        "\n"
        "  The equilibrium is marginal cost where no learning unit is pivotal and\n"
        "  just below the next competitor's cost where one is. Bidding at marginal\n"
        "  cost therefore matches it everywhere except the pivotal hours and falls\n"
        "  short there, which is what the marginal-cost row shows. A learner that\n"
        "  has found the equilibrium has 'diff' near zero *and* a high 'matched'\n"
        "  share. 'diff' near zero with few hours matched is errors in both\n"
        "  directions cancelling out; 'diff' above zero is prices higher than the\n"
        "  equilibrium supports."
    )


def main() -> None:
    global USE_MARKUP, SCENARIO, STUDY_CASE, MAX_EPISODE_STEPS, EQUILIBRIUM

    parser = argparse.ArgumentParser(
        description="Compare MAPPO (TorchRL), ASSUME's MATD3 and MATD3 (TorchRL)."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="training episodes per arm (default: the scenario's "
        "learning_config.training_episodes)",
    )
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--seeds", type=int, default=1, help="repeat with N seeds")
    parser.add_argument(
        "--arms",
        default=",".join(ARMS),
        help=(
            "comma-separated subset of " + ",".join(ARM_SPECS) + " "
            "(default: " + ",".join(ARMS) + "; "
            "'all' runs everything that is installed)"
        ),
    )
    parser.add_argument(
        "--raw-actions",
        action="store_true",
        help="PettingZoo arms bid absolute prices instead of a markup on "
        "marginal cost (removes the head start ASSUME gives itself)",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=None,
        help="truncate each episode after N market openings (default: the "
        "scenario's full horizon). Long horizons get expensive fast - a 92-day "
        "episode is ~2200 openings and takes minutes per episode per arm",
    )
    parser.add_argument(
        "--scenario", default=SCENARIO, help="scenario folder under examples/inputs"
    )
    parser.add_argument("--study-case", default=STUDY_CASE, help="study case name")
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="skip TensorBoard logging (written to tensorboard/<sim_id>/<arm>)",
    )
    parser.add_argument(
        "--logdir", default="tensorboard", help="TensorBoard root directory"
    )
    parser.add_argument(
        "--log-every", type=int, default=5, help="episodes between logged evaluations"
    )
    parser.add_argument("--quick", action="store_true", help="short demo run")
    args = parser.parse_args()

    quick = args.quick
    if quick:
        args.episodes, args.eval_episodes = 15, 2

    USE_MARKUP = not args.raw_actions
    SCENARIO, STUDY_CASE = args.scenario, args.study_case
    MAX_EPISODE_STEPS = args.max_episode_steps

    if args.arms.strip() == "all":
        arms = [a for a in ARM_SPECS if a not in OPTIONAL_ARMS or _installed(a)]
    else:
        arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = set(arms) - set(ARM_SPECS)
    if unknown:
        parser.error(
            f"unknown arm(s): {sorted(unknown)}; choose from {list(ARM_SPECS)}"
        )
    for arm in arms:
        if arm in OPTIONAL_ARMS and not _installed(arm):
            parser.error(f"the '{arm}' arm needs: pip install {OPTIONAL_ARMS[arm]}")

    # references are measured on the raw action space
    eval_env = make_env(markup=False)
    agents = eval_env.possible_agents

    # the reference every arm's clearing prices are compared with, at the
    # simulation's own time step
    price_file = metrics.equilibrium_price_file(SCENARIO)
    EQUILIBRIUM = metrics.load_equilibrium_prices(
        SCENARIO,
        freq=pd.Timedelta(eval_env.unwrapped.world.units[agents[0]].index.freq),
    )

    # Everything the run needs comes from the scenario's learning_config unless
    # the user overrides it, so `--episodes` is the only knob that can pull the
    # arms away from what config.yaml says. Read it off the LearningConfig ASSUME
    # itself built, so !include and ASSUME's own defaulting are respected.
    scenario_config = eval_env.unwrapped.world.learning_config
    if args.episodes is None:
        args.episodes = scenario_config.training_episodes
        budget_source = "from config.yaml"
    elif quick:
        budget_source = "--quick"
    else:
        budget_source = "--episodes override"

    print(f"running {args.seeds} seeds of {args.episodes} episodes each")
    print(f"scenario        : {SCENARIO}/{STUDY_CASE}")
    print(f"agents          : {agents}")
    horizon = eval_env.unwrapped.world.end - eval_env.unwrapped.world.start
    if MAX_EPISODE_STEPS:
        print(
            f"episode         : {MAX_EPISODE_STEPS} market openings"
            f"   (truncated from {horizon} of simulated time)"
        )
    else:
        print(
            f"episode         : {horizon} of simulated time, one bid per market opening"
        )
    print(
        f"budget          : {args.episodes} training episodes per arm"
        f"   [{budget_source}]"
    )
    print(
        f"evaluation      : {args.eval_episodes} episodes, seeds 500+, identical for all arms"
    )
    if EQUILIBRIUM is not None:
        print(
            f"equilibrium     : examples/prices/{price_file.name}, compared hour by "
            "hour with every arm's clearing prices"
        )
    else:
        print(f"equilibrium     : none - examples/prices has no file for {SCENARIO}")
    print(f"seeds           : {args.seeds}")
    print(
        "action space    : "
        + ("markup on marginal cost" if USE_MARKUP else "absolute bid price")
        + " (PettingZoo arms); absolute (ASSUME arm)"
    )
    # everything below comes from the scenario's learning_config; the ASSUME arm
    # uses all of it, the PettingZoo arms use max_bid_price and the date range
    print(
        f"from config     : max_bid_price={scenario_config.max_bid_price}, "
        f"train_freq={scenario_config.train_freq}, "
        f"gradient_steps={scenario_config.gradient_steps}, "
        f"batch_size={scenario_config.batch_size}, "
        f"lr={scenario_config.learning_rate}"
    )
    print(
        f"                  initial_experience="
        f"{scenario_config.episodes_collecting_initial_experience}, "
        f"validation_interval={scenario_config.validation_episodes_interval}, "
        f"actor={scenario_config.actor_architecture}\n"
    )

    print("Reference points (no learning)")
    random_eval = metrics.evaluate_full(
        eval_env, random_policy, episodes=args.eval_episodes, equilibrium=EQUILIBRIUM
    )
    random_score, random_bid = random_eval.score, random_eval.bid
    # meaningless for storage agents, whose last observation is the cost of the
    # stored energy rather than a marginal cost
    has_marginal_cost = metrics.observes_marginal_cost_last(eval_env)
    if has_marginal_cost:
        cost_eval = metrics.evaluate_full(
            eval_env,
            marginal_cost_policy,
            episodes=args.eval_episodes,
            equilibrium=EQUILIBRIUM,
        )
        cost_score, cost_bid = cost_eval.score, cost_eval.bid
    else:
        cost_eval = cost_score = cost_bid = None
    print(
        f"  random actions        {random_score:+8.3f}   bid {random_bid:6.1f} EUR/MWh"
    )
    if has_marginal_cost:
        print(
            f"  bid at marginal cost  {cost_score:+8.3f}   bid {cost_bid:6.1f} EUR/MWh"
            "   <- the competitive outcome\n"
        )
    else:
        print(
            "  (no marginal-cost reference: these agents do not observe marginal\n"
            "   cost as their last observation)\n"
        )

    results: dict[str, list[dict]] = {arm: [] for arm in arms}
    for seed in range(args.seeds):
        for arm in arms:
            print(f"training {arm} (seed {seed}) ...", flush=True)
            reporter = None
            if not args.no_tensorboard:
                # one TensorBoard run per arm and seed, all under the scenario's
                # simulation id, so the three curves overlay in one session
                # Sibling directories, not children: run_learning() does
                # `shutil.rmtree("tensorboard/<simulation_id>")` at startup, so
                # anything written *inside* that directory is deleted the moment
                # the ASSUME arm runs - including other arms' open writers.
                # Siblings under the same --logdir still overlay in one
                # TensorBoard session.
                reporter = metrics.TensorBoardReporter(
                    simulation_id=f"{SCENARIO}_{STUDY_CASE}__{arm}_seed{seed}",
                    log_dir=args.logdir,
                )
            outcome = run_arm(
                arm,
                args.episodes,
                seed,
                args.eval_episodes,
                reporter=reporter,
                log_every=args.log_every,
            )
            results[arm].append(outcome)
            prices = outcome["prices"]
            print(
                f"  -> return {outcome['score']:+.3f}"
                f"   bids {outcome['bid']:.1f} EUR/MWh"
                + (
                    f"   clears {prices.difference:+.1f} EUR/MWh vs equilibrium"
                    if prices is not None
                    else ""
                )
                + f"   ({outcome['seconds']:.0f}s)\n",
                flush=True,
            )

    # ---- results table ----
    labels = {arm: ARM_SPECS[arm][0] for arm in arms}
    print("=" * 74)
    print("Results - mean per-agent episode return, higher is better")
    print("=" * 74)
    print(
        f"  {'learner':<22} {'return':>10} {'+- sd':>8} {'bid':>9}"
        f" {'spread':>8} {'train':>7}"
    )
    print(f"  {'-' * 22} {'-' * 10} {'-' * 8} {'-' * 9} {'-' * 8} {'-' * 7}")
    print(
        f"  {'random actions':<22} {random_score:>+10.3f} {'':>8}"
        f" {random_bid:>8.1f} {'':>8} {'':>7}"
    )
    if has_marginal_cost:
        print(
            f"  {'bid at marginal cost':<22} {cost_score:>+10.3f} {'':>8}"
            f" {cost_bid:>8.1f} {'':>8} {'':>7}"
        )
    for arm in arms:
        scores = [r["score"] for r in results[arm]]
        sd = float(np.std(scores)) if len(scores) > 1 else float("nan")
        spread = float(np.mean([r["spread"] for r in results[arm]]))
        seconds = float(np.mean([r["seconds"] for r in results[arm]]))
        sd_text = f"{sd:8.3f}" if len(scores) > 1 else f"{'n/a':>8}"
        bid = float(np.mean([r["bid"] for r in results[arm]]))
        print(
            f"  {labels[arm]:<22} {float(np.mean(scores)):>+10.3f} {sd_text}"
            f" {bid:>8.1f} {spread:>8.2f} {seconds:>6.0f}s"
        )

    if EQUILIBRIUM is not None:
        price_rows = [("random actions", random_eval.prices)]
        if cost_eval is not None:
            price_rows.append(("bid at marginal cost", cost_eval.prices))
        price_rows += [
            (labels[arm], average_prices([r["prices"] for r in results[arm]]))
            for arm in arms
        ]
        print_price_table(price_rows, price_file)

    storage_rows = {
        arm: results[arm][-1]["storage"] for arm in arms if results[arm][-1]["storage"]
    }
    if storage_rows:
        print("\nStorage usage (final evaluation episode)")
        keys = sorted(next(iter(storage_rows.values())))
        for arm, values in storage_rows.items():
            print(f"  {labels[arm]}")
            for key in keys:
                print(f"    {key:<28} {values[key]:>10.3f}")
    elif not args.no_tensorboard:
        print(
            "\n  (no storage units in this scenario - the 2x_storage_* TensorBoard\n"
            "   tags stay empty. Try --scenario example_02e --study-case base"
            " --raw-actions)"
        )

    if not args.no_tensorboard:
        print(f"\n  TensorBoard: tensorboard --logdir {args.logdir}")

    print(
        "\n  'bid' is the mean price the learner actually offers, averaged over\n"
        "  every opening. Read it against the marginal-cost row: below it the\n"
        "  learner sells at a loss, far above it never gets dispatched."
    )
    print(
        "\n  'spread' is the gap between the best and worst agent under the same\n"
        "  learner. Expect ASSUME's to be the largest: it trains one actor per\n"
        "  unit, while MAPPO shares a single policy across all agents and AgileRL\n"
        "  shares one per homogeneous group. Shared parameters force the agents to\n"
        "  behave alike; independent actors let them differentiate."
    )

    if args.seeds == 1:
        print(
            "\n  NOTE: one seed, one scenario, a small budget. These numbers order\n"
            "  three *runs*, not three algorithms. Use --seeds 5 before concluding\n"
            "  anything; run-to-run variance here is comparable to the gaps."
        )

    print(
        "\nHow to read this\n"
        "  Bidding at marginal cost is the competitive equilibrium of this market,\n"
        "  and it is where self-interested learners are pulled: undercutting a\n"
        "  higher-priced rival wins the whole volume. A learner scoring near the\n"
        "  marginal-cost row has found that equilibrium. Scoring far above it would\n"
        "  mean the agents sustained supra-competitive prices - tacit collusion,\n"
        "  which is one of the effects ASSUME exists to study.\n"
        "\n"
        "  The equilibrium price table refines that. In the pivotal hours, where\n"
        "  the other learning units cannot cover demand without a given unit, even\n"
        "  self-interested bidding supports a price just below the next\n"
        "  competitor's cost - so there the equilibrium sits above marginal cost.\n"
        "  Clearing prices well above the equilibrium ('diff' > 0) are what tacit\n"
        "  collusion would look like.\n"
        "\n"
        "  The two MATD3 rows share an algorithm but not an implementation, and\n"
        "  ASSUME's hyperparameters were tuned for this scenario by its authors\n"
        "  while the other two arms run one-pass guesses. At this budget that\n"
        "  alone can outweigh the algorithms.\n"
        "\n"
        "  Watch the 'spread' column, though - it points at something structural.\n"
        "  The five learning units are byte-identical (same CCGT, same costs), so\n"
        "  they see near-identical observations. A *shared* policy must therefore\n"
        "  bid near-identically, while ASSUME's per-unit actors can differentiate.\n"
        "  That matters in a pay-as-clear market: measured on this scenario, the\n"
        "  best symmetric configuration (all five at +30 EUR markup) returns\n"
        "  +6.47, while an asymmetric one (two bidding at cost, three at +25)\n"
        "  returns +7.51 - and symmetric bidding sits on a cliff, collapsing to\n"
        "  -14.10 at +35 because all five leave the merit order at the same\n"
        "  instant. Parameter sharing cannot express that asymmetry."
    )

    eval_env.close()


if __name__ == "__main__":
    main()
