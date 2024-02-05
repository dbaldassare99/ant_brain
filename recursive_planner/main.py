import retro
from algo import RP
from buffer import TorchGym
import numpy as np
import torch

torch.random.manual_seed(0)
torch.set_printoptions(precision=2, sci_mode=False)


env = retro.make(
    game="SuperMarioBros-Nes", use_restricted_actions=retro.Actions.DISCRETE
)
env = TorchGym(env)
player = RP(env)
player.play()
