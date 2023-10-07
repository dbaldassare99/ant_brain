# import gymnasium as gym
from buffer import Timestep, ExperienceBuffer, TorchGym
import torch
import retro
from gymnasium.utils.play import play
import numpy as np


class Human:
    def __init__(self, env):
        self.env = env
        self.notes = ExperienceBuffer()

    def save(
        self,
        obs_t,
        obs_tp1,
        action,
        rew: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ):
        pass
        # obs_t = torch.tensor(obs_t).float()
        # self.notes.add(
        #     Timestep(
        #         obs=obs_t,
        #         act=action,
        #         rew=rew,
        #         terminated=terminated,
        #         truncated=truncated,
        #         info=info,
        #     )
        # )

    def play_human(self):
        play(
            self.env,
            zoom=4,
            callback=self.save,
            keys_to_action={
                "c": 3,  # left
                "g": 7,  # right
                "r": 18,  # jump
                "u": 21,  # jump left
                "y": 25,  # jump right
            },
            noop=0,
        )


if __name__ == "__main__":
    env = retro.make(
        game="SuperMarioBros-Nes",
        use_restricted_actions=retro.Actions.DISCRETE,
        render_mode="rgb_array",
    )
    # env = TorchGym(env)
    player = Human(env)
    player.play_human()
    player.notes.save("human_play.pkl")
