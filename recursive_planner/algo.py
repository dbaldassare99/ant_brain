# Get Goal
# Recursive bad boy:
from agent import Brain
from newton import optimize_subplan
import torch
from buffer import ExperienceBuffer, StateAction


class RP:
    def __init__(self, env) -> None:
        self.env = env
        self.net = Brain()
        self.notes = ExperienceBuffer()

    def recursive_actor(
        self, obs: torch.Tensor, goal: torch.Tensor, noise: torch.Tensor
    ):
        # Put obs, goal into net
        optimized_noise = optimize_subplan(self.net, obs, goal, noise)
        ret = self.net(obs, goal, optimized_noise)

        # if obs → general goal < 0.5 # safeguard
        if ret.gen_poss < 0.5:
            # Return failure
            return "failure"
        # If can act >=90%
        if ret.this_turn_poss >= 0.9:
            # Get action
            action_sequence = ret.get_action_sequence()
            # Do actions
            for action in action_sequence[0]:  # not parallel
                self.env.step(action)
            # Check action
            optimized_noise = optimize_subplan(self.net, obs, goal, noise)
            ret = self.net(obs, goal, optimized_noise)
            good_action = 1 if ret.predicted_moves < 3 else 0
            good_plan = 1 if good_action == 1 else None
            self.notes.add(StateAction(obs, goal, ret, good_plan, good_action))
            # Take action note
            # Return check plan

            # Else
            # Get Midpoint
            # Take plan note of Call with (obs, midpoint)
            # If this one fails, still try the next one…
            # Take plan note and Return Call with (current state, goal)
