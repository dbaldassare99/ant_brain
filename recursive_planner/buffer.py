import torch
from agent import Brain
from torch.func import jacrev, vmap


class StateQueue:
    def __init__(self):
        self.obs_list = []
        self.reward_total = 0
        self.terminated = False

    def __call__(self, env_step):
        obs, reward, terminated, truncated, info = env_step
        self.obs_list.append(obs)
        self.reward_total += reward
        self.terminated = terminated

    def midpoint(self):
        return self.obs_list[len(self.obs_list) // 2]

    def final_obs(self):
        return self.obs_list[-1]


class State:
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
        num_moves: torch.Tensor,
        noise: torch.Tensor,
    ):
        self.obs = obs
        self.goal = goal
        self.vects = vects
        self.gen_poss = gen_poss
        self.this_turn_poss = this_turn_poss
        self.midpoint = midpoint
        self.predicted_reward = predicted_reward
        self.acts = acts
        self.num_moves = num_moves
        self.noise = noise

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

    def goal_optim_loss(self):
        return self.predicted_reward - self.num_moves + self.gen_poss

    def subplan_optim_loss(self):
        return (
            self.predicted_reward - self.num_moves + self.this_turn_poss + self.gen_poss
        )

    # works batched
    def optimize_subplan(
        self,
        net: Brain,
        steps: int = 100,
    ):
        net = net.eval()
        jacobian = vmap(
            jacrev(
                lambda x, y, z: net.arg_fwd_unbatched(x, y, z).subplan_optim_loss(), 2
            )
        )
        for _ in range(steps):
            grad = jacobian(net, self.goal, self.noise)
            grad = grad.squeeze(1)
            self.noise = self.noise + grad

    # no batch!
    def optimize_goal(
        self,
        net: Brain,
    ):
        net = net.eval()
        jacobian = jacrev(
            lambda x, y, z: self.arg_fwd_unbatched(x, y, z).goal_optim_loss(), (1, 2)
        )
        while self.gen_poss < 0.8:
            grads = jacobian(net, self.goal, self.noise)
            grads = [g.squeeze(1) for g in grads]
            self.goal = self.goal + grads[1]
            self.noise = self.noise + grads[2]
            self.forward(net)

    def arg_fwd_unbatched(self, net, goal, noise):
        ins = [self.obs, goal, noise]
        ins = [x.unsqueeze(0) for x in ins]
        outs = net(ins)
        outs = [x.squeeze(0) for x in outs]
        (
            self.gen_poss,
            self.this_turn_poss,
            self.midpoint,
            self.acts,
            self.num_moves,
        ) = outs
        return self

    def forward(self, net: Brain):
        ins = [self.obs, self.goal, self.noise]
        if len(self.obs.shape) == 3:  # batch unbatched
            ins = [x.unsqueeze(0) for x in ins]
            post = lambda xs: [x.squeeze(0) for x in xs]  # re-unbatch
        else:
            post = lambda xs: xs
        (
            self.gen_poss,
            self.this_turn_poss,
            self.midpoint,
            self.acts,
            self.num_moves,
        ) = post(net(ins))


class Memory:
    def __init__(
        self,
        state: State,
    ):
        self.obs = state.obs
        self.goal = state.goal
        self.vects = state.vects
        self.gen_poss = state.gen_poss
        self.this_turn_poss = state.this_turn_poss
        self.midpoint = state.midpoint
        self.reward = state.predicted_reward
        self.acts = state.acts
        self.num_moves = state.num_moves
        self.noise = state.noise


class ActionMemory(Memory):
    def __init__(
        self, state: State, queue: StateQueue, action_sequence: list[torch.Tensor]
    ) -> None:
        super().__init__(state)
        self.can_act = 1 if state.num_moves < 3 else 0
        self.good_plan = 1 if self.can_act == 1 else None
        self.midpoint = queue.midpoint()
        self.num_moves = len(action_sequence)
        self.last_obs = queue.final_obs()


class PlanMemory(Memory):
    def __init__(self, state: State, memories: list[ActionMemory]) -> None:
        super().__init__(state)
        self.num_moves = sum([m.num_moves for m in memories])
        self.predicted_reward = sum([m.reward for m in memories])
        self.obs = memories[0].obs
        self.goal = memories[-1].goal
        # we could say no... because we're alrealy < 95% this turn poss to get here
        self.this_turn_poss = None

    def add(self, memory: Memory) -> None:
        self.num_moves += memory.num_moves
        self.predicted_reward += memory.reward
        self.goal = memory.goal


class ExperienceBuffer:
    def __init__(self) -> None:
        self.buffer = []

    def add(self, example: State):
        self.buffer.append(example)
