import torch
from nets import Brain, BrainOut
from torch.func import jacrev, vmap
import numpy as np
from dataclasses import dataclass
from tqdm import trange
from typing import List

# class Buffer:
#     def __init__(self) -> None:
#         self.buffer = []

#     def add(self, example):
#         self.buffer.append(example)


class SMA:
    def __init__(self, length: int):
        self.length = length
        self.buffer = []
        self.total = 0

    def add(self, x: float):
        self.buffer.append(x)
        self.total += x
        if len(self.buffer) > self.length:
            self.total -= self.buffer.pop(0)

    def mean(self):
        return self.total / len(self.buffer)


@dataclass
class Timestep:
    obs: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict
    act: int


class StateQueue:
    def __init__(self):
        self.obs_list = []
        self.reward_total = 0
        self.terminated = False

    def __call__(self, env_step):
        obs, reward, terminated, truncated, info = env_step
        self.obs_list.append(obs)
        self.reward_total += reward
        self.terminated = terminated

    def midpoint(self):
        return self.obs_list[len(self.obs_list) // 2]

    def final_obs(self):
        return self.obs_list[-1]


class State:
    def __init__(
        self,
        obs: torch.Tensor = None,
        goal: torch.Tensor = torch.randn(16),  # honestly, too jank 4 me
        noise: torch.Tensor = torch.randn(16),  # yep that's jank
        vects: torch.Tensor = None,
        midpoint: torch.Tensor = None,
        acts: torch.Tensor = None,
        gen_poss: torch.Tensor = 0,
        poss_this_turn: torch.Tensor = 0,
        rew: torch.Tensor = 0,
        num_moves: torch.Tensor = 0,
    ):
        self.obs = obs
        self.goal = goal
        self.vects = vects
        self.gen_poss = gen_poss
        self.poss_this_turn = poss_this_turn
        self.midpoint = midpoint
        self.rew = rew
        self.acts = acts
        self.num_moves = num_moves
        self.noise = noise
        self.lr = 0.5

    def get_action_sequence(self) -> list:
        if len(self.acts.shape) < 3:  # if unbatched
            acts = self.acts.unsqueeze(0)
        acts = torch.argmax(acts, dim=1)
        indices = torch.where(acts == 36, acts, 0.0)
        indices = indices.nonzero()
        action_list = [*torch.split(acts, 1)]
        action_list = [a.squeeze() for a in action_list]
        visited = []
        for batch, idx in indices:
            if batch in visited:
                continue
            visited.append(batch)
            action_list[batch] = action_list[batch][:idx]
        return action_list

    # no_batch
    def optimize_subplan(
        self,
        net: Brain,
        steps: int = 100,
    ):
        def subplan_optim_loss(xs):
            outs = BrainOut(xs)
            # outs.norm_scalars()
            return outs.poss_this_turn  # + outs.rew  # - outs.num_moves

        self.noise = torch.randn_like(self.noise)
        self.update(net)
        # print(self.poss_this_turn)
        net = net.eval()
        jacobian = jacrev(lambda x, y, z: subplan_optim_loss(net(x, y, z)), 2)
        before = subplan_optim_loss(net(self.obs, self.goal, self.noise))
        for _ in trange(steps, desc=f"subplan_optim"):
            grad = jacobian(self.obs, self.goal, self.noise)
            grad = grad.squeeze(0)
            self.noise = self.noise + grad * self.lr
        after = subplan_optim_loss(net(self.obs, self.goal, self.noise))
        self.update(net)
        # print(self.poss_this_turn)
        print(before, after)
        print(after - before)
        return before, after

    # no batch!
    def optimize_plan(
        self,
        net: Brain,
        steps: int = 200,
    ):
        def gen_poss_loss(xs):
            outs = BrainOut(xs)
            # outs.norm_scalars()
            return outs.gen_poss

        def full_plan_loss(xs):
            outs = BrainOut(xs)
            # outs.norm_scalars()
            return outs.gen_poss + outs.rew  # - outs.num_moves

        self.noise = torch.randn_like(self.noise)
        self.goal = torch.randn_like(self.goal)
        net = net.eval()
        gen_poss_grad = jacrev(lambda x, y, z: gen_poss_loss(net(x, y, z)), (1, 2))
        full_opt = jacrev(lambda x, y, z: full_plan_loss(net(x, y, z)), (1, 2))
        before = gen_poss_loss(net(self.obs, self.goal, self.noise))
        self.update(net)
        pbar = trange(steps, desc=f"plan_optim")
        for i in pbar:
            if self.gen_poss > 0.5:
                grads = full_opt(self.obs, self.goal, self.noise)
            else:
                grads = gen_poss_grad(self.obs, self.goal, self.noise)
            grads = [g.squeeze(0) for g in grads]
            self.goal = self.goal + grads[0] * self.lr
            self.noise = self.noise + grads[1] * self.lr
            self.update(net)
            pbar.set_postfix_str(f"gen_poss: {self.gen_poss.item():.2f}")
        after = gen_poss_loss(net(self.obs, self.goal, self.noise))
        return before, after

    def update(
        self,
        net: Brain,
        obs: torch.Tensor = None,
        goal: torch.Tensor = None,
        noise: torch.Tensor = None,
    ) -> None:
        net.eval()
        self.obs = obs if obs else self.obs
        self.goal = goal if goal else self.goal
        self.noise = noise if noise else self.noise

        outs = BrainOut(net(self.obs, self.goal, self.noise))
        self.gen_poss = outs.gen_poss
        self.poss_this_turn = outs.poss_this_turn
        self.midpoint = outs.midpoint
        self.acts = outs.acts
        self.num_moves = outs.num_moves
        self.rew = outs.rew


class Memory:
    def __init__(
        self,
        state: State,
    ):
        self.obs = state.obs
        self.goal = state.goal
        self.vects = state.vects
        self.gen_poss = state.gen_poss
        self.poss_this_turn = state.poss_this_turn
        self.midpoint = state.midpoint
        self.rew = state.rew
        self.acts = state.acts
        self.num_moves = state.num_moves
        self.noise = state.noise

    def add_action(
        self, queue: StateQueue, action_sequence: list[torch.Tensor]
    ) -> None:
        self.can_act = 1 if self.num_moves < 3 else 0
        self.good_plan = 1 if self.can_act == 1 else None
        self.midpoint = queue.midpoint()
        self.num_moves = len(action_sequence)
        self.last_obs = queue.final_obs()

    def add_plan(self, memories: list) -> None:
        self.num_moves = sum([m.num_moves for m in memories])
        self.rew = sum([m.reward for m in memories])
        self.obs = memories[0].obs
        self.goal = memories[-1].goal
        # we could say no... because we're alrealy < 95% this turn poss to get here
        self.poss_this_turn = None

    # def add(self, memory: Memory) -> None: # maybe need this later?
    #     self.num_moves += memory.num_moves
    #     self.predicted_reward += memory.reward
    #     self.goal = memory.goal

    def rand_start(self, buffer: list[Timestep], start: int, end: int):
        self.obs = buffer[start].obs
        self.goal = buffer[end].obs
        self.midpoint = buffer[(start + end) // 2].obs
        self.num_moves = end - start
        self.rew = sum([r.reward for r in buffer[start:end]])
        self.acts = torch.tensor([t.act for t in buffer[start : start + 10]])
        if self.num_moves < 10:
            self.acts[self.num_moves] = 36
        self.noise = None
        self.gen_poss = 1 if self.num_moves <= 50 else 0
        self.poss_this_turn = 1 if self.num_moves <= 10 else 0
        return self


class ExperienceBuffer:
    def __init__(self) -> None:
        self.buffer: List[Memory] = []
        # self.num_moves_norm = torch.nn.BatchNorm1d(1)
        # self.rew_norm = torch.nn.BatchNorm1d(1)

    def add(self, example: Memory):
        self.buffer.append(example)

    def sample_preprocess(self, net: Brain, batch_size: int):
        mems = [
            self.buffer[i] for i in np.random.randint(0, len(self.buffer), batch_size)
        ]
        obs = torch.stack([m.obs for m in mems]).float()
        goal = torch.stack([m.goal for m in mems]).float()
        if len(goal[0].shape) > 2:
            goal = net.vision(net.preprocess_frame(goal))
        noise = torch.rand_like(goal).float()
        ins = (obs, goal, noise)

        midpoint = torch.stack([m.midpoint for m in mems])
        if len(midpoint[0].shape) > 2:
            midpoint = net.vision(net.preprocess_frame(midpoint))
        acts = torch.stack([m.acts for m in mems]).long()

        def scalar_prep(x):
            return torch.tensor(x).unsqueeze(-1).float()

        rew = scalar_prep([m.rew for m in mems])
        num_moves = scalar_prep([m.num_moves for m in mems])
        poss_this_turn = scalar_prep([m.poss_this_turn for m in mems])
        gen_poss = scalar_prep([m.gen_poss for m in mems])
        # num_moves = num_moves.unsqueeze(1)
        # num_moves = self.num_moves_norm(num_moves) # doesn't help
        # num_moves = num_moves.squeeze(1)
        # label_dict = {
        #     "gen_poss": gen_poss,
        #     "poss_this_turn": poss_this_turn,
        #     "midpoint": midpoint,
        #     "acts": acts,
        #     "num_moves": num_moves,
        #     "rew": rew,
        # }
        labels = BrainOut(
            {
                "gen_poss": gen_poss,
                "poss_this_turn": poss_this_turn,
                "midpoint": midpoint,
                "acts": acts,
                "num_moves": num_moves,
                "rew": rew,
            }
        )
        return ins, labels


class TorchGym:
    def __init__(self, gym):
        self.gym = gym
        self.action_space = gym.action_space

    # gymnasium.Env.step(self, action: ActType) → tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]
    def step(self, action):
        ret = self.gym.step(action)
        obs = torch.tensor(ret[0]).float()
        rew = torch.tensor(ret[1]).float()
        return obs, rew, ret[2], ret[3], ret[4]

    # gymnasium.Env.reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) → tuple[ObsType, dict[str, Any]]
    def reset(self):
        ret = self.gym.reset()
        obs = torch.tensor(ret[0]).float()
        return obs, ret[1]
