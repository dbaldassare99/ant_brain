import retro
from algo import RP
from buffer import TorchGym


env = retro.make(
    game="SuperMarioBros-Nes", use_restricted_actions=retro.Actions.DISCRETE
)
env = TorchGym(env)
player = RP(env)
player.play()
