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
        obs: torch.Tensor = None,
        goal: torch.Tensor = torch.randn(224, 240, 3),  # honestly, too jank 4 me
        noise: torch.Tensor = torch.randn(16),  # yep that's jank
        vects: torch.Tensor = None,
        midpoint: torch.Tensor = None,
        acts: torch.Tensor = None,
        gen_poss: torch.Tensor = 0,
        this_turn_poss: torch.Tensor = 0,
        predicted_reward: torch.Tensor = 0,
        num_moves: torch.Tensor = 0,
    ):
        self.obs = obs
        self.goal = goal
        self.vects = vects
        self.generally_possible = gen_poss
        self.possible_this_turn = this_turn_poss
        self.midpoint = midpoint
        self.predicted_reward = predicted_reward
        self.acts = acts
        self.num_moves = num_moves
        self.noise = noise
        self.lr = 5

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

    # def goal_optim_loss(self):
    #     return self.predicted_reward - self.num_moves + self.gen_poss

    # def subplan_optim_loss(self):
    #     return (
    #         self.predicted_reward - self.num_moves + self.this_turn_poss + self.gen_poss
    #     )

    # no_batch
    def optimize_subplan(
        self,
        net: Brain,
        steps: int = 50,
    ):
        def subplan_optim_loss(xs):
            return xs[5] - xs[4] + xs[1] + xs[0]

        net = net.eval()
        jacobian = jacrev(lambda x, y, z: subplan_optim_loss(net(x, y, z)), 2)
        before = subplan_optim_loss(net(self.obs, self.goal, self.noise))
        for _ in range(steps):
            grad = jacobian(self.obs, self.goal, self.noise)
            grad = grad.squeeze(0)
            self.noise = self.noise + grad * self.lr
        after = subplan_optim_loss(net(self.obs, self.goal, self.noise))
        self.update(net)
        return before, after

    # no batch!
    def optimize_plan(
        self,
        net: Brain,
        steps: int = 200,
    ):
        def plan_optim_loss(xs):
            return xs[5] - xs[4] + xs[0]

        net = net.eval()
        jacobian = jacrev(lambda x, y, z: plan_optim_loss(net(x, y, z)), (1, 2))
        count = 0
        before = plan_optim_loss(net(self.obs, self.goal, self.noise))
        while self.generally_possible < 0.9 and count < steps:
            count += 1
            grads = jacobian(self.obs, self.goal, self.noise)
            grads = [g.squeeze(0) for g in grads]
            self.goal = self.goal + grads[0] * self.lr
            self.noise = self.noise + grads[1] * self.lr
            self.update(net)
        after = plan_optim_loss(net(self.obs, self.goal, self.noise))
        return before, after

    def update(
        self,
        net: Brain,
        obs: torch.Tensor = None,
        goal: torch.Tensor = None,
        noise: torch.Tensor = None,
    ) -> None:
        net.eval()
        self.obs = obs if obs else self.obs
        self.goal = goal if goal else self.goal
        self.noise = noise if noise else self.noise

        (
            self.generally_possible,
            self.possible_this_turn,
            self.midpoint,
            self.acts,
            self.num_moves,
            self.predicted_reward,
        ) = net(self.obs, self.goal, self.noise)


class Memory:
    def __init__(
        self,
        state: State,
    ):
        self.obs = state.obs
        self.goal = state.goal
        self.vects = state.vects
        self.gen_poss = state.generally_possible
        self.this_turn_poss = state.possible_this_turn
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
