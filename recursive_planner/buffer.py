import torch


class StateAction:
    def __init__(
        self,
        obs: torch.Tensor,
        goal: torch.Tensor,
        vects: torch.Tensor,
        gen_poss: torch.Tensor,
        this_turn_poss: torch.Tensor,
        midpoint: torch.Tensor,
        acts: torch.Tensor,
        predicted_reward: torch.Tensor,
        predicted_moves: torch.Tensor,
    ):
        self.obs = obs
        self.goal = goal
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

    def squeeze_0(self):
        self.vects = self.vects.squeeze(0)
        self.gen_poss = self.gen_poss.squeeze(0)
        self.this_turn_poss = self.this_turn_poss.squeeze(0)
        self.midpoint = self.midpoint.squeeze(0)
        self.predicted_reward = self.predicted_reward.squeeze(0)
        self.predicted_moves = self.predicted_moves.squeeze(0)
        self.acts = self.acts.squeeze(0)
        return self

    def get_action_sequence(self) -> list:
        assert len(self.acts.shape) == 3  # batched
        acts = torch.argmax(self.acts, dim=-1)
        indices = torch.where(acts == 32, acts, 0.0)
        indices = indices.nonzero()
        action_list = [*torch.split(acts, 1)]
        action_list = [a.squeeze() for a in action_list]
        visited = []
        for batch, idx in indices:
            if batch in visited:
                continue
            visited.append(batch)
            action_list[batch] = action_list[batch][:idx]
        return action_list

    # Get Goal:
    # Grad on good plan, reward and short sequence
    # Update noise and goal_vect
    # Return goal_vect
    def goal_optim_loss(self):
        return self.predicted_reward - self.predicted_moves + self.gen_poss

    # Get subplan:
    # Grad on good plan, can act, reward, and short sequence
    # Update noise
    # Return midpoint
    def subplan_optim_loss(self):
        return (
            self.predicted_reward
            - self.predicted_moves
            + self.this_turn_poss
            + self.gen_poss
        )


class ExperienceBuffer:
    def __init__(self) -> None:
        self.buffer = []

    def add(self, example: StateAction):
        self.buffer.append(example)
