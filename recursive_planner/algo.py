# Get Goal
# Recursive bad boy:
from agent import Brain
from newton import optimize_subplan, optimize_goal
import torch
from buffer import ExperienceBuffer, State, StateQueue, ActionMemory, PlanMemory


class RP:
    def __init__(self, env) -> None:
        self.env = env
        self.net = Brain()
        self.notes = ExperienceBuffer()

    def play(self):
        obs, reward, terminated, truncated, info = self.env.reset()
        while True:
            goal, noise = self.make_goal(obs)
            self.recursive_actor(obs, goal, noise)

    def make_goal(self, obs):
        goal, noise = optimize_goal(self.net, obs, torch.randn(16))

    def recursive_actor(
        self,
        state: State,
    ):
        if state.this_turn_poss >= 0.95:
            action_sequence = state.get_action_sequence()
            queue = StateQueue()
            for action in action_sequence[0]:  # not parallel
                queue(self.env.step(action))
            state.optimize_subplan(self.net)
            self.notes.add(ActionMemory(state, queue, action_sequence))
            return (
                len(action_sequence),
                queue.final_obs(),
                queue.reward_total,
            )
        else:
            state.optimize_subplan(self.net)
            state_1 = self.do_plan(state)
            state_2 = self.do_plan(state)
            # check
            rew_so_far = rew_1 + rew_2
            sub_noise = optimize_subplan(self.net, obs, goal, noise)
            ret = self.net(obs, goal, sub_noise)
            if ret.num_moves < 3 or expected_reward - rew_so_far <= 3:
                return "success?"
            else:

            num_actions = num_actions_1 + num_actions_2
            rew = rew_1 + rew_2
            return num_actions, obs, rew

    def do_plan(self, state):
        self.recursive_actor(state)
        self.notes.add(PlanMemory)
        return state
