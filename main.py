import gymnasium as gym
import tianshou as ts
import torch, numpy as np
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
import retro

# why is mario stuck going left?


ENV_STRING = "CartPole-v0"
ENV_STRING = "highway-v0"
ENV_STRING = "SuperMarioBros-Nes"


env = retro.make(game=ENV_STRING, use_restricted_actions=retro.Actions.DISCRETE)
# env = gym.make(ENV_STRING)

# train_envs = ts.env.SubprocVectorEnv([lambda: retro.make(ENV_STRING) for _ in range(100)])
# test_envs = ts.env.SubprocVectorEnv([lambda: retro.make(ENV_STRING) for _ in range(10)])


class Net(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(np.prod(state_shape), 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, np.prod(action_shape)),
        )

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        return logits, state


state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n
print(f"state shape: {state_shape}, action shape: {action_shape}")
net = Net(state_shape, action_shape)
optim = torch.optim.Adam(net.parameters(), lr=1e-3)

env.close()

train_envs = ts.env.SubprocVectorEnv(
    [
        lambda: retro.make(ENV_STRING, use_restricted_actions=retro.Actions.DISCRETE)
        for _ in range(2)
    ]
)
test_envs = ts.env.SubprocVectorEnv(
    [
        lambda: retro.make(ENV_STRING, use_restricted_actions=retro.Actions.DISCRETE)
        for _ in range(2)
    ]
)


policy = ts.policy.DQNPolicy(
    net, optim, discount_factor=0.9, estimation_step=3, target_update_freq=320
)

train_collector = ts.data.Collector(
    policy, train_envs, ts.data.VectorReplayBuffer(200, 10), exploration_noise=True
)
test_collector = ts.data.Collector(policy, test_envs, exploration_noise=True)


writer = SummaryWriter("log/dqn")
logger = TensorboardLogger(writer)

print("Start training")
# env.spec.reward_threshold = 2_000
# print(f"reward thresh = {env.spec.reward_threshold}")
result = ts.trainer.offpolicy_trainer(
    policy,
    train_collector,
    test_collector,
    max_epoch=10,
    step_per_epoch=1000,
    step_per_collect=10,
    update_per_step=0.1,
    episode_per_test=100,
    batch_size=64,
    train_fn=lambda epoch, env_step: policy.set_eps(0.1),
    test_fn=lambda epoch, env_step: policy.set_eps(0.05),
    stop_fn=lambda mean_rewards: mean_rewards >= 200,
    logger=logger,
)
print(f'Finished training! Use {result["duration"]}')


torch.save(policy.state_dict(), "dqn.pth")
policy.load_state_dict(torch.load("dqn.pth"))

render_env = retro.make(ENV_STRING, render_mode="human")
policy.eval()
policy.set_eps(0.05)
collector = ts.data.Collector(policy, render_env, exploration_noise=True)
collector.collect(n_episode=1, render=1 / 35)
