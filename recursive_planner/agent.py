import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


# convolutional network that takes in a frame of shape (224, 240, 3)
# and outputs a vector of shape (1, 16)
class Vision(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=16, stride=6),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=8, stride=3),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.BatchNorm1d(16),
        )
        self.norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def forward(self, obs):
        obs = self.norm(obs)
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
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.attn = transformer_encoder

    def forward(self, seen):
        return self.attn(seen)


class Vect2Scalar(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(48, 16),
            nn.ReLU(),
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

    def forward(self, vects):
        batch = vects.shape[0]
        concatted = vects.view(batch, 48)
        return self.model(concatted)


class Action(nn.Module):
    def __init__(self):
        super().__init__()
        self.unfold = nn.LSTM(1, 16, batch_first=True)
        self.encoder_h = Vects2_16()
        self.encoder_c = Vects2_16()
        self.action_decoder = nn.Linear(16, 33)

    def forward(self, vects):
        batch = vects.shape[0]
        # concatted = vects.view(batch, 48)
        h = self.encoder_h(vects).unsqueeze(0)
        c = self.encoder_c(vects).unsqueeze(0)
        inputs = torch.ones(batch, 10, 1)
        acts, (h, c) = self.unfold(inputs, (h, c))
        acts = self.action_decoder(acts)
        return acts


class Brain(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = Vision()
        self.trunk = Trunk()
        self.generally_possibe = Vect2Scalar()
        self.possibe_this_turn = Vect2Scalar()
        self.predicted_moves = Vect2Scalar()
        self.pre_reward = Vect2Scalar()
        self.midpoint = Vects2_16()
        self.acts = Action()

    def preprocess(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        obs = obs.squeeze()
        batch = obs.shape[0]
        obs = obs.view(batch, 3, 224, 240)
        return obs

    def see(self, obs):
        # obs = torch.split(obs, 1, dim=1)
        seen = [self.vision(self.preprocess(ob)) for ob in obs]
        seen = torch.stack(seen, dim=1)
        return seen

    def forward(self, frame_1, frame_2, noise):
        obs = [frame_1, frame_2]
        seen = self.see(obs)
        seen = torch.cat([seen, noise.unsqueeze(1)], dim=1)
        vects = self.trunk(seen)
        output = BrainOutput(
            vects,
            self.generally_possibe(vects),
            this_turn_poss=self.possibe_this_turn(vects),
            midpoint=self.midpoint(vects),
            acts=self.acts(vects),
            predicted_reward=self.pre_reward(vects),
            predicted_moves=self.predicted_moves(vects),
        )
        return output


class BrainOutput:
    def __init__(
        self,
        vects,
        gen_poss,
        this_turn_poss,
        midpoint,
        acts,
        predicted_reward,
        predicted_moves,
    ):
        self.vects = vects
        self.gen_poss = gen_poss
        self.this_turn_poss = this_turn_poss
        self.midpoint = midpoint
        self.predicted_reward = predicted_reward
        self.predicted_moves = predicted_moves
        self.acts = acts

    def __str__(self):
        return f"""
        BrainOutput
        vects: {self.vects.shape}
        gen_poss: {self.gen_poss.shape}
        this_turn_poss: {self.this_turn_poss.shape}
        mid: {self.midpoint.shape}
        predicted reward: {self.predicted_reward.shape}
        predicted moves: {self.predicted_moves.shape}
        acts: {self.acts.shape}
        """

    # Get Goal:
    # Grad on good plan, reward and short sequence
    # Update noise and goal_vect
    # Return goal_vect

    # Get midpoint:
    # Grad on good plan, can act, reward, and short sequence
    # Update noise
    # Return midpoint

    # Get action:
    # Grad on can act, short sequence, reward, good plan x times?
    # Update noise
    # Return actions

    def action_optim_loss(self):
        return (
            self.predicted_reward
            - self.predicted_moves
            + self.this_turn_poss
            + self.gen_poss
        )
