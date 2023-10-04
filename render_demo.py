# import gymnasium as gym
# import sys
# import time
# import retro


# import retro

# for game in retro.data.list_games():
#     print(game, retro.data.list_states(game))

# env = retro.make(
#     game="Airstriker-Genesis",
#     render_mode="human",
#     use_restricted_actions=retro.Actions.DISCRETE,
# )

# # env_string = "DeepmindLabSeekavoidArena01-v0"
# # env = gym.make(env_string, render_mode="human")
# env.reset()
# env.render()
# time.sleep(200)

import retro


ENV_STRING = "SuperMarioBros-Nes"


def main():
    env = retro.make(game=ENV_STRING, use_restricted_actions=retro.Actions.DISCRETE)
    env.reset()
    reward_sum = 0

    action_shape = env.action_space.shape or env.action_space.n
    while True:
        obs, reward, terminated, truncated, info = env.step(6)
        reward_sum += reward
        print(reward_sum)
        env.render()
        if terminated:
            env.reset()
    env.close()


if __name__ == "__main__":
    main()
