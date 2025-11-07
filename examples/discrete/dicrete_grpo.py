from typing import TypeVar
import gymnasium as gym
import numpy as np
import torch
import tianshou as ts
from torch.utils.tensorboard import SummaryWriter

from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.env.venvs import InitialStateStrategy, StateSettableWrapper
from tianshou.algorithm.modelfree.reinforce import DiscreteActorPolicy
from tianshou.data import CollectStats, VectorReplayBuffer
from tianshou.trainer import OnPolicyTrainerParams
from tianshou.utils.net.common import Net
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


def main() -> None:
    task = "CartPole-v1"
    lr, epoch, batch_size = 1e-3, 10, 64
    max_batchsize = 256
    eps_clip, kl_coef = 0.2, 0.1
    num_training_envs, num_test_envs = 10, 100
    update_step_num_repetitions = 10
    buffer_size = 20000
    epoch_num_steps, collection_step_num_env_steps = 10000, 2000
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger = ts.utils.TensorboardLogger(
        SummaryWriter("log/grpo_discrete")
    )  # TensorBoard is supported!

    train_envs = ts.env.DummyVectorEnv(
        [
            lambda initial_state=initial_state: StateSettableWrapper[
                np.ndarray, np.ndarray, np.ndarray
            ](
                gym.make("CartPole-v1"),
                InitialStateStrategyCardPole(initial_state),
            )
            for initial_state in initial_states_array
        ]
    )
    test_envs = ts.env.DummyVectorEnv([lambda: gym.make(task) for _ in range(num_test_envs)])

    env = gym.make(task, render_mode="human")
    assert isinstance(env.action_space, gym.spaces.Discrete)
    space_info = SpaceInfo.from_env(env)
    state_shape = space_info.observation_info.obs_shape
    action_shape = space_info.action_info.action_shape

    net = Net(state_shape=state_shape, hidden_sizes=[128, 128])
    actor = DiscreteActor(preprocess_net=net, action_shape=action_shape).to(device)
    dist = torch.distributions.Categorical

    optim = AdamOptimizerFactory(lr=lr)

    policy = DiscreteActorPolicy(
        actor=actor,
        dist_fn=dist,
        action_space=env.action_space,
        deterministic_eval=True,
    )

    algorithm = ts.algorithm.GRPO(
        policy=policy,
        optim=optim,
        max_batchsize=max_batchsize,
        clip_epsilon=eps_clip,
        kl_coefficient=kl_coef,
    )

    train_collector = ts.data.Collector[CollectStats](
        algorithm,
        train_envs,
        VectorReplayBuffer(buffer_size, len(train_envs)),
    )

    test_collector = ts.data.Collector[CollectStats](algorithm, test_envs)

    def stop_fn(mean_rewards: float) -> bool:
        if env.spec:
            if not env.spec.reward_threshold:
                return False
            else:
                return mean_rewards >= env.spec.reward_threshold
        return False

    result = algorithm.run_training(
        OnPolicyTrainerParams(
            train_collector=train_collector,
            test_collector=test_collector,
            max_epochs=epoch,
            epoch_num_steps=epoch_num_steps,
            update_step_num_repetitions=update_step_num_repetitions,
            test_step_num_episodes=num_test_envs,
            batch_size=batch_size,
            collection_step_num_env_steps=collection_step_num_env_steps,
            stop_fn=stop_fn,
            logger=logger,
            test_in_train=True,
        )
    )

    print(f"Finished training in {result.timing.total_time} seconds")

    # watch performance
    collector = ts.data.Collector[CollectStats](algorithm, env, exploration_noise=True)
    collector.reset_env()
    collector.collect(n_episode=1000, render=1 / 35)


if __name__ == "__main__":
    main()
