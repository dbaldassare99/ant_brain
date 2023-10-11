import torch
from nets import VisionTrainer, BrainOut, Brain, VECT_LEN
from torch.func import jacrev, vmap
import numpy as np
from dataclasses import dataclass
from tqdm import trange
from typing import List
import matplotlib.pyplot as plt
from torchvision.transforms.functional import rgb_to_grayscale
from torch.utils.data import Dataset


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
    rew: float
    terminated: bool
    truncated: bool
    info: dict
    act: int


class ExperienceDataset(Dataset):
    def __init__(self, buffer: List[Timestep], net: Brain) -> None:
        self.buffer = buffer
        self.net = net

    def __getitem__(self, idx: int) -> Timestep:
        curr = self.buffer[idx]
        next = self.buffer[idx + 1]
        return (curr.obs, next.obs), (curr.act, next.rew)

    def __len__(self) -> int:
        return len(self.buffer) - 1


class ExperienceBuffer:
    def __init__(self):
        self.buffer: List[Timestep] = []
        self.sma = SMA(100)

    def add(self, timestep: Timestep):
        self.buffer.append(timestep)
        self.sma.add(timestep.rew)

    def __getitem__(self, idx: int) -> Timestep:
        return self.buffer[idx]

    def to_Dataset(self, net: Brain) -> ExperienceDataset:
        return ExperienceDataset(self.buffer, net)

    def vision_examples(self, batch_size: int, make_fig: bool = False):
        sample_idxs = np.random.randint(0, len(self.buffer) - 1, batch_size)
        frame_1 = torch.stack([self.buffer[i].obs for i in sample_idxs])
        frame_2 = torch.stack([self.buffer[i + 1].obs for i in sample_idxs])
        act = torch.tensor([self.buffer[i].act for i in sample_idxs])
        rew = torch.tensor([self.buffer[i + 1].rew for i in sample_idxs])
        if make_fig:
            fig, axes = plt.subplots(2, batch_size, figsize=(30, 7))
            for i in range(batch_size):
                axes[0, i].imshow(frame_1[i].int())
                axes[1, i].imshow(frame_2[i].int())
            label_dict = {
                "acts": act,
                "rew": rew,
            }
            for ax, i in zip(axes[0], range(batch_size)):
                infos = [f"{k}: {v[i].item():.2f}" for k, v in label_dict.items()]
                infos = str.join("\n", infos)
                ax.set_title(infos)
            fig.tight_layout()
            plt.savefig("test.png")
        return (frame_1, frame_2), (act, rew)


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
        goal: torch.Tensor = torch.randn(VECT_LEN),  # honestly, too jank 4 me
        noise: torch.Tensor = torch.randn(VECT_LEN),  # yep that's jank
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
        self.lr = 2

    def get_action_sequence(self) -> list:
        acts = self.acts
        if len(acts.shape) < 3:  # if unbatched
            acts = acts.unsqueeze(0)
        acts = torch.argmax(acts, dim=1)
        indices = torch.where(acts == 6, acts, 0.0)
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
        steps: int = 200,
    ):
        def subplan_optim_loss(xs):
            outs = BrainOut(xs)
            return outs.poss_this_turn + outs.rew  # - outs.num_moves

        def update_str():
            poss_this_turn = f"poss_this_turn: {self.poss_this_turn.item():.2f}"
            rew = f"rew:{self.rew.item():.2f}"
            gen_poss = f"gen_poss:{self.gen_poss.item():.2f}"
            return str.join(", ", [poss_this_turn, rew, gen_poss])

        encoded_obs = net.vision_encode(self.obs)
        self.noise = torch.randn_like(self.noise)
        net = net.eval()
        jacobian = jacrev(
            lambda x, y, z: subplan_optim_loss(net.forward_without_vision(x, y, z)), 2
        )
        self.update(net)
        before = subplan_optim_loss(net(self.obs, self.goal, self.noise))
        before_str = update_str()
        # pbar = trange(steps, desc=f"subplan_optim")
        # for i in pbar:
        for i in range(steps):
            grad = jacobian(encoded_obs, self.goal, self.noise)
            grad = grad.squeeze(0)
            self.noise = self.noise + grad * self.lr
            # self.update(net)
            # pbar.set_postfix_str(
            #     f"poss_this_turn: {self.poss_this_turn.item():.2f} | rew: {self.rew.item():.2f}"
            # )
        self.update(net)
        after = subplan_optim_loss(net(self.obs, self.goal, self.noise))
        after_str = update_str()
        print("BEFORE", before_str, "AFTER", after_str)
        return before, after

    # no batch!
    def optimize_plan(
        self,
        net: Brain,
        steps: int = 400,
    ):
        def gen_poss_loss(xs):
            outs = BrainOut(xs)
            # outs.norm_scalars()
            return outs.gen_poss

        def full_plan_loss(xs):
            outs = BrainOut(xs)
            # outs.norm_scalars()
            return outs.gen_poss + outs.rew  # - outs.num_moves

        def update_str():
            gen_poss = f"gen_poss: {self.gen_poss.item():.2f}"
            rew = f"rew:{self.rew.item():.2f}"
            num_moves = f"num_moves:{self.num_moves.item():.2f}"
            return str.join(", ", [gen_poss, rew, num_moves])

        encoded_obs = net.vision_encode(self.obs)
        self.noise = torch.randn_like(self.noise)
        self.goal = torch.randn_like(self.goal)
        net = net.eval()
        # gen_poss_grad = jacrev(lambda x, y, z: gen_poss_loss(net(x, y, z)), (1, 2))
        full_opt = jacrev(
            lambda x, y, z: full_plan_loss(net.forward_without_vision(x, y, z)), (1, 2)
        )
        self.update(net)
        before = gen_poss_loss(net(self.obs, self.goal, self.noise))
        before_str = update_str()
        for i in range(steps):
            grads = full_opt(encoded_obs, self.goal, self.noise)
            grads = [g.squeeze(0) for g in grads]
            self.goal = self.goal + grads[0] * self.lr
            self.noise = self.noise + grads[1] * self.lr
        self.update(net)
        after = gen_poss_loss(net(self.obs, self.goal, self.noise))
        after_str = update_str()
        print("BEFORE", before_str, "AFTER", after_str)
        return before, after

    def update(
        self,
        net: VisionTrainer,
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
        self.learn_from = {}
        self.last_obs = self.obs

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
        self.rew = sum([m.rew for m in memories])
        self.obs = memories[0].obs
        self.goal = memories[-1].goal
        self.last_obs = memories[-1].last_obs
        # we could say no... because we're alrealy < 95% this turn poss to get here
        self.poss_this_turn = None

    def get_grey(self, frames):
        # frame = [rgb_to_grayscale(frame.view(3, 224, 240)) for frame in frames]
        frame = rgb_to_grayscale(frames[0].permute(2, 1, 0)).permute(2, 1, 0)
        # print(frames[0].shape)
        plt.imshow(frame)
        plt.savefig("test.png")
        print(frame.shape)
        assert 2 == 3
        frame = torch.stack(frame, 0)
        frame = frame.view(224, 240, 3)
        return frame

    def rand_start(self, buffer: list[Timestep], start: int, end: int):
        # self.obs = self.get_grey([buffer[i].obs for i in range(start, start + 3)])
        # self.goal = self.get_grey([buffer[i].obs for i in range(end, end + 3)])
        self.obs = buffer[start].obs
        # print(self.obs.shape)
        # assert 2 == 3
        self.goal = buffer[end].obs

        self.midpoint = buffer[(start + end) // 2].obs
        self.num_moves = end - start
        self.rew = sum([r.rew for r in buffer[start:end]])
        self.acts = torch.tensor([t.act for t in buffer[start : start + 10]])
        if self.num_moves < 10:
            self.acts[self.num_moves] = 6
        self.learn_from["acts"] = self.num_moves < 15
        self.noise = None
        self.gen_poss = 1 if self.num_moves <= 100 else 0
        self.poss_this_turn = 1 if self.num_moves <= 10 else 0
        return self


class MemoryDataset(Dataset):
    def __init__(self, buffer: List[Memory], net: Brain) -> None:
        self.buffer = buffer
        self.net = net

    def __getitem__(self, idx: int) -> Memory:
        mem = self.buffer[idx]
        goal = mem.goal

        if len(goal.shape) > 2:
            with torch.no_grad():
                goal = self.net.vision_encode(goal)
        noise = torch.rand_like(goal)
        ins = {"obs": mem.obs, "goal": goal, "noise": noise}

        midpoint = mem.midpoint
        if len(midpoint.shape) > 2:
            with torch.no_grad():
                midpoint = self.net.vision_encode(midpoint)
        labels = {
            "gen_poss": torch.tensor(mem.gen_poss).unsqueeze(-1).float(),
            "poss_this_turn": torch.tensor(mem.poss_this_turn).unsqueeze(-1).float(),
            "midpoint": midpoint,
            "acts": mem.acts,
            "num_moves": torch.tensor(mem.num_moves).unsqueeze(-1).float(),
            "rew": mem.rew.unsqueeze(-1).float(),
        }
        return ins, labels

    def __len__(self) -> int:
        return len(self.buffer)


class MemoryBuffer:
    def __init__(self) -> None:
        self.buffer: List[Memory] = []
        self.num_moves_norm = torch.nn.BatchNorm1d(1)
        self.rew_norm = torch.nn.BatchNorm1d(1)

    def add(self, example: Memory):
        self.buffer.append(example)

    def to_Dataset(self, net: Brain) -> MemoryDataset:
        return MemoryDataset(self.buffer, net)

    def sample_preprocess_and_batch(
        self, net: Brain, batch_size: int, print: bool = False
    ):
        mems = np.random.choice(self.buffer, batch_size)
        obs = torch.stack([m.obs for m in mems]).float()
        goal = torch.stack([m.goal for m in mems]).float()
        if print:
            fig, axes = plt.subplots(2, batch_size, figsize=(30, 7))
            for i in range(batch_size):
                axes[0, i].imshow(obs[i].int())
                axes[1, i].imshow(goal[i].int())

        if len(goal[0].shape) > 2:
            goal = net.vision(net.preprocess_frame(goal))
        noise = torch.ones_like(goal).float()
        ins = (obs, goal, noise)

        midpoint = torch.stack([m.midpoint for m in mems])
        if len(midpoint[0].shape) > 2:
            midpoint = net.vision(net.preprocess_frame(midpoint))
        acts = torch.stack([m.acts for m in mems]).long()
        learn_from = [m.learn_from["acts"] for m in mems]

        def scalar_prep(x):
            return torch.tensor(x).unsqueeze(-1).float()

        rew = scalar_prep([m.rew for m in mems])
        num_moves = scalar_prep([m.num_moves for m in mems])
        poss_this_turn = scalar_prep([m.poss_this_turn for m in mems])
        gen_poss = scalar_prep([m.gen_poss for m in mems])
        # num_moves = num_moves.unsqueeze(1)
        # num_moves = self.num_moves_norm(num_moves)
        # num_moves = num_moves.squeeze(1)
        # rew = rew.unsqueeze(1)
        # rew = self.rew_norm(rew)
        # rew = rew.squeeze(1)
        if print:
            label_dict = {
                "gen_poss": gen_poss,
                "poss_this_turn": poss_this_turn,
                "midpoint": midpoint,
                "acts": acts,
                "num_moves": num_moves,
                "rew": rew,
            }
            for ax, i in zip(axes[0], range(batch_size)):
                infos = [
                    f"{k}: {v[i].item():.2f}"
                    for k, v in label_dict.items()
                    if k != "acts" and k != "midpoint"
                ]
                infos = str.join("\n", infos)
                ax.set_title(infos)
            fig.tight_layout()
            plt.savefig("test.png")

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
        return ins, labels, learn_from


# class MemoryDataset:
#     def __init__(self, buffer, idxs) -> None:
#         self.buffer = buffer


class TorchGym:
    def __init__(self, gym):
        self.gym = gym
        self.action_space = gym.action_space
        self.frame_avg = torch.zeros(224, 240, 3)

    def rand_act(self):
        return np.random.randint(0, 6)

    # gymnasium.Env.step(self, action: ActType) → tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]
    def step(self, action):
        action_dict = {
            0: 16,  # noop
            1: 18,  # 3,  # left lol only right plz
            2: 7,  # right
            3: 18,  # jump
            4: 21,  # 21,  # jump left
            5: 25,  # jump right
        }
        if isinstance(action, torch.Tensor):
            action = action.item()
        action = action_dict[action]
        ret = self.gym.step(action)
        self.frame_avg = torch.roll(self.frame_avg, -1, 2)
        obs = torch.tensor(ret[0]).float()
        self.frame_avg[:, :, -1] = rgb_to_grayscale(obs.permute(2, 0, 1)).squeeze()
        rew = torch.tensor(ret[1]).float()
        # return self.frame_avg, rew, ret[2], ret[3], ret[4]
        return obs, rew, ret[2], ret[3], ret[4]

    # gymnasium.Env.reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) → tuple[ObsType, dict[str, Any]]
    def reset(self):
        ret = self.gym.reset()
        obs = torch.tensor(ret[0]).float()
        return obs, ret[1]
