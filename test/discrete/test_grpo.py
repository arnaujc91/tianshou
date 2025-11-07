import argparse
import os
from typing import TypeVar

from test.determinism_test import AlgorithmDeterminismTest

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tianshou.env.venvs import InitialStateStrategy, StateSettableWrapper
from tianshou.algorithm.modelfree.grpo import GRPO
from tianshou.algorithm.algorithm_base import Algorithm
from tianshou.algorithm.modelfree.reinforce import DiscreteActorPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import Collector, CollectStats, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.trainer import OnPolicyTrainerParams
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import (
    ActionReprNet,
    ActionReprNetDataParallelWrapper,
    Net,
)
from tianshou.utils.net.discrete import DiscreteActor
from tianshou.utils.space_info import SpaceInfo


initial_states = [
    [0.1, 0.2, 0.05, 0.1],  # State for env 1
    [0.0, 0.0, 0.0, 0.0],  # State for env 2
    [-0.1, 0.1, -0.05, 0.2],  # State for env 3
]

initial_states_array = np.array(initial_states)

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
StateType = TypeVar("StateType")

class InitialStateStrategyCardPole(InitialStateStrategy[ObsType, StateType]):
    def __init__(self, initial_state: StateType) -> None:
        self.initial_state = initial_state

    def compute_initial_state(self, env: gym.Env) -> StateType:
        return self.initial_state

    def set_state(self, env: gym.Env, state: StateType) -> None:
        env.unwrapped.state = state

    def compute_obs_from_state(self, env: gym.Env, state: StateType) -> ObsType:
        return np.array(env.unwrapped.state, dtype=np.float32)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="CartPole-v1")
    parser.add_argument("--reward_threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1626)
    parser.add_argument("--buffer_size", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--epoch_num_steps", type=int, default=50000)
    parser.add_argument("--collection_step_num_env_steps", type=int, default=2000)
    parser.add_argument("--update_step_num_repetitions", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_sizes", type=int, nargs="*", default=[64, 64])
    parser.add_argument("--num_train_envs", type=int, default=3)
    parser.add_argument("--num_test_envs", type=int, default=100)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    # grpo special
    parser.add_argument("--eps_clip", type=float, default=0.2)
    parser.add_argument("--kl_coef", type=float, default=0.1)
    parser.add_argument("--max_batchsize", type=int, default=256)
    return parser.parse_known_args()[0]


def test_grpo(args: argparse.Namespace = get_args(), enable_assertions: bool = True) -> None:
    env = gym.make(args.task)
    space_info = SpaceInfo.from_env(env)
    args.state_shape = space_info.observation_info.obs_shape
    args.action_shape = space_info.action_info.action_shape
    if args.reward_threshold is None:
        default_reward_threshold = {"CartPole-v1": 195}
        args.reward_threshold = default_reward_threshold.get(
            args.task,
            env.spec.reward_threshold if env.spec else None,
        )

    # Create training environments with fixed initial states for GRPO
    train_envs = DummyVectorEnv(
        [
            lambda initial_state=initial_state: StateSettableWrapper[np.ndarray, np.ndarray, np.ndarray](
                gym.make("CartPole-v1"),
                InitialStateStrategyCardPole(initial_state),
            )
            for initial_state in initial_states_array
        ]
    )
    test_envs = DummyVectorEnv([lambda: gym.make(args.task) for _ in range(args.num_test_envs)])

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # model - GRPO only needs actor, no critic
    net = Net(state_shape=args.state_shape, hidden_sizes=args.hidden_sizes)
    actor: ActionReprNet
    if torch.cuda.is_available():
        actor = ActionReprNetDataParallelWrapper(
            DiscreteActor(preprocess_net=net, action_shape=args.action_shape).to(args.device)
        )
    else:
        actor = DiscreteActor(preprocess_net=net, action_shape=args.action_shape).to(args.device)

    # orthogonal initialization
    for m in actor.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)

    optim = AdamOptimizerFactory(lr=args.lr)
    dist = torch.distributions.Categorical
    policy = DiscreteActorPolicy(
        actor=actor,
        dist_fn=dist,
        action_space=env.action_space,
        deterministic_eval=True,
    )

    # Create GRPO algorithm
    algorithm: GRPO = GRPO(
        policy=policy,
        optim=optim,
        max_batchsize=args.max_batchsize,
        clip_epsilon=args.eps_clip,
        kl_coefficient=args.kl_coef,
    )

    # collector
    train_collector = Collector[CollectStats](
        algorithm,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
    )
    test_collector = Collector[CollectStats](algorithm, test_envs)

    # log
    log_path = os.path.join(args.logdir, args.task, "grpo")
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    def save_best_fn(policy: Algorithm) -> None:
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def stop_fn(mean_rewards: float) -> bool:
        return mean_rewards >= args.reward_threshold

    # trainer
    result = algorithm.run_training(
        OnPolicyTrainerParams(
            train_collector=train_collector,
            test_collector=test_collector,
            max_epochs=args.epoch,
            epoch_num_steps=args.epoch_num_steps,
            update_step_num_repetitions=args.update_step_num_repetitions,
            test_step_num_episodes=args.num_test_envs,
            batch_size=args.batch_size,
            collection_step_num_env_steps=args.collection_step_num_env_steps,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            logger=logger,
            test_in_train=True,
        )
    )

    if enable_assertions:
        assert stop_fn(result.best_reward)


def test_grpo_determinism() -> None:
    main_fn = lambda args: test_grpo(args, enable_assertions=False)
    AlgorithmDeterminismTest("discrete_grpo", main_fn, get_args()).run()


if __name__ == "__main__":
    test_grpo()
