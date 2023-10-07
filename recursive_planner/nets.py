from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from typing_extensions import Self
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer


# convolutional network that takes in a frame of shape (224, 240, 3)
# and outputs a vector of shape (1, 16)
class Vision(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=16, stride=6),
            # nn.ReLU(),
            nn.Sigmoid(),
            nn.Conv2d(16, 16, kernel_size=8, stride=3),
            # nn.ReLU(),
            nn.Sigmoid(),
            nn.Conv2d(16, 16, kernel_size=4, stride=2),
            # nn.ReLU(),
            nn.Sigmoid(),
            nn.Conv2d(16, 16, kernel_size=4, stride=2),
            # nn.ReLU(),
            nn.Sigmoid(),
            nn.Flatten(),
            # nn.BatchNorm1d(16),
        )
        self.norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def forward(self, obs):
        # obs = self.norm(obs)
        vect = self.model(obs)
        return vect


# This needs positional encoding!!
class Trunk(nn.Module):
    def __init__(self):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=16,
            nhead=8,
            dim_feedforward=32,
            batch_first=True,
        )
        self.attn = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.pos_enc = Summer(PositionalEncoding1D(16))

    def forward(self, seen):
        seen = self.pos_enc(seen)
        return self.attn(seen)


class Vect2Scalar(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(48, 16),
            # nn.ReLU(),
            nn.Sigmoid(),
            nn.Linear(16, 1),
        )

    def forward(self, vects):
        batch = vects.shape[0]
        concatted = vects.view(batch, 48)
        return self.model(concatted)


class Vects2_16(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, 16),
        )

    def forward(self, vects) -> torch.Tensor:
        batch = vects.shape[0]
        concatted = vects.view(batch, 48)
        return self.model(concatted)


class Action(nn.Module):
    def __init__(self):
        super().__init__()
        # self.unfold = nn.LSTM(1, 16, batch_first=True)
        # self.encoder_h = Vects2_16()
        # self.encoder_c = Vects2_16()
        # self.action_decoder = nn.Linear(16, 37)

        # 48 + 37 + 1 = 85
        # self.unfold = nn.Linear(85, 37)
        self.nets = [nn.Linear(48, 7) for _ in range(10)]
        self.pre_nets = nn.Sequential(
            nn.Linear(48, 48),
            nn.ReLU(),
            nn.Linear(48, 48),
            nn.ReLU(),
            nn.Linear(48, 48),
            nn.ReLU(),
        )

    def forward(self, vects: torch.Tensor):
        # batch = vects.shape[0]
        # vects = vects.view(batch, 48)
        # ins = torch.ones(batch, 37)
        # outs = torch.zeros(batch, 37, 10)
        # for i in range(10):
        #     ins = torch.cat([ins, vects], dim=1)
        #     ins = self.unfold(ins)
        #     ins = torch.nn.functional.softmax(ins, dim=1)
        #     outs[:, :, i] = ins
        # return outs

        # batch = vects.shape[0]
        # h = self.encoder_h(vects).unsqueeze(0)
        # c = self.encoder_c(vects).unsqueeze(0)
        # inputs = torch.ones(batch, 10, 1)
        # acts, (h, c) = self.unfold(inputs, (h, c))
        # acts = self.action_decoder(acts)
        # acts = acts.view(batch, 37, 10)
        # return acts

        batch = vects.shape[0]
        vects = vects.view(batch, 48)
        # vects = self.pre_nets(vects)
        acts = [layer(vects) for layer in self.nets]
        return torch.stack(acts, dim=2)


class Brain(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = Vision()
        self.trunk = Trunk()
        self.generally_possibe = Vect2Scalar()
        self.possibe_this_turn = Vect2Scalar()
        self.num_moves = Vect2Scalar()
        self.predicted_reward = Vect2Scalar()
        self.midpoint = Vects2_16()
        self.acts = Action()

    def preprocess_frame(self, ob):
        batch = ob.shape[0]
        ob = ob.view(batch, 3, 224, 240)
        return ob

    def forward(
        self, frame_1: torch.Tensor, goal: torch.Tensor, noise: torch.Tensor
    ) -> dict:
        batched = True
        if len(frame_1.shape) == 3:  # batch if not batched
            frame_1 = frame_1.unsqueeze(0)
            goal = goal.unsqueeze(0)
            noise = noise.unsqueeze(0)
            batched = False
        obs = [frame_1]  # add in 2nd frame here later to see goal
        seen = [self.vision(self.preprocess_frame(ob)) for ob in obs]
        seen = torch.stack(seen, dim=1)
        seen = torch.cat([seen, goal.unsqueeze(1), noise.unsqueeze(1)], dim=1)
        vects = seen
        # vects = self.trunk(seen)  # doesn't help with loss
        rets = {
            "gen_poss": F.sigmoid(self.generally_possibe(vects)),
            "poss_this_turn": F.sigmoid(self.possibe_this_turn(vects)),
            "midpoint": self.midpoint(vects),
            "acts": self.acts(vects),
            "num_moves": self.num_moves(vects),
            "rew": self.predicted_reward(vects),
        }
        if not batched:  # unbatch if not batched
            rets = {k: v.squeeze(0) for k, v in rets.items()}
        return rets


class BrainOut:
    def __init__(self, outs: dict[str, torch.Tensor]):
        self.gen_poss = outs["gen_poss"]
        self.poss_this_turn = outs["poss_this_turn"]
        self.midpoint = outs["midpoint"]
        self.acts = outs["acts"]
        self.num_moves = outs["num_moves"]
        self.rew = outs["rew"]
        # self.scalar_normalizer = Normalizer(4)

    def __iter__(self):
        return iter(
            [
                self.gen_poss,
                self.poss_this_turn,
                self.midpoint,
                self.acts,
                self.num_moves,
                self.rew,
            ]
        )

    # slightly cursed copilot code
    def replace_if_none(self, other: Self):
        for k, v in self.__dict__.items():
            if v is None:
                self.__dict__[k] = other.__dict__[k]


#     def norm_scalars(self):
#         normalized = torch.stack(
#             [self.gen_poss, self.poss_this_turn, self.num_moves, self.rew]
#         ).view(4)
#         normalized = self.scalar_normalizer(normalized)
#         (
#             self.gen_poss,
#             self.poss_this_turn,
#             self.num_moves,
#             self.rew,
#         ) = normalized.split(1)


# class Normalizer:
#     def __init__(self, channels: int, maxlen: int = 1_000):
#         self.maxlen = maxlen
#         self.channels = channels
#         self.q = torch.zeros(maxlen, channels)
#         self.idx = 0

#     def __call__(self, vector: torch.Tensor):
#         self.add(vector)
#         return (vector - self.mean()) / self.std() + 0.5

#     def add(self, item: torch.Tensor):
#         self.q[self.idx] = item
#         self.idx += 1
#         if self.idx >= self.maxlen:
#             self.idx = 0

#     def mean(self):
#         return self.q.mean(dim=0)

#     def std(self):
#         return self.q.std(dim=0)
