# Get Goal
# Recursive bad boy:
from agent import Brain
from newton import optimize_subplan, optimize_goal
import torch
from buffer import ExperienceBuffer, State, StateQueue, ActionMemory, PlanMemory, Memory
import copy


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
    ) -> Memory:
        arg_state = copy.copy(state)  # to see how close we got!
        state.optimize_subplan(self.net)
        if state.this_turn_poss >= 0.95:
            action_sequence = state.get_action_sequence()
            queue = StateQueue()
            for action in action_sequence[0]:  # not parallel
                queue(self.env.step(action))
            # get info abt the action
            state.optimize_subplan(self.net)
            action_record = ActionMemory(state, queue, action_sequence)
            self.notes.add(action_record)
            return action_record
        else:
            # first half
            state_part_1 = copy.copy(state)
            state_part_1.goal = state.midpoint
            memory_part_1 = self.recursive_actor(state_part_1)
            # second half
            state_part_2 = copy.copy(state)
            state_part_2.obs = memory_part_1.last_obs
            memory_part_2 = self.recursive_actor(state_part_2)
            # third half (lol) (really 2nd half again)
            state_part_3 = copy.copy(state)
            state_part_3.obs = memory_part_2.last_obs
            memory_part_3 = self.recursive_actor(state_part_3)

            # get info abt the plan
            check_state = copy.copy(state)
            check_state.obs = memory_part_3.last_obs
            check_state.optimize_subplan(self.net)
            plan_record = PlanMemory(
                state, [memory_part_1, memory_part_2, memory_part_3]
            )
            plan_record.gen_poss = (
                1
                if check_state.num_moves < 3
                or arg_state.predicted_reward - plan_record.prediced_reward <= 3
                else 0
            )
            self.notes.add(plan_record)
            return plan_record
