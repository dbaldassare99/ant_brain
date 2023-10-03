import retro
from algo import RP


env = retro.make(
    game="SuperMarioBros-Nes", use_restricted_actions=retro.Actions.DISCRETE
)
player = RP(env)
player.play()
