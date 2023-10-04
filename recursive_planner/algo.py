# Get Goal
# Recursive bad boy:
from nets import Brain
import numpy as np
import torch
from buffer import (
    ExperienceBuffer,
    State,
    StateQueue,
    Memory,
    Timestep,
)
import copy


class RP:
    def __init__(self, env) -> None:
        self.env = env
        self.net = Brain()
        self.notes = ExperienceBuffer()

    def play(self):
        self.random_bootstrap()
        obs, info = self.env.reset()
        state = State(obs=obs)

        while True:
            state.optimize_plan(self.net)
            memory = self.recursive_actor(state)
            state.obs = memory.last_obs

    def random_bootstrap(self) -> None:
        buffer = []
        _ = self.env.reset()
        random_play_len = 1_000
        max_sample_length = 100
        assert max_sample_length < random_play_len
        for _ in range(random_play_len):
            action = self.env.action_space.sample()
            buffer.append(Timestep(*self.env.step(action), action))

        for _ in range(100):
            start = np.random.randint(0, random_play_len - max_sample_length)
            end = start + np.random.randint(1, max_sample_length)
            mem = Memory(State()).rand_start(buffer, start, end)
            print(mem)
            self.notes.add(mem)

        self.train()

    def recursive_actor(
        self,
        state: State,
    ) -> Memory:
        arg_state = copy.copy(state)  # to see how close we got!
        state.optimize_subplan(self.net)
        if state.possible_this_turn >= 0.5:
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

    def train(self):
        optimizer = torch.optim.Adam(self.net.parameters())
        for _ in range(100):
            ins, labels = self.notes.sample_preprocess(self.net, 10)
            optimizer.zero_grad()
            outputs = self.net(ins)
            loss = self.loss_fn(outputs, labels)
            print(loss)
            loss.backward()
            optimizer.step()

    def loss_fn(self, outputs, labels):
        def scalar_only(l):
            return (l[0], l[1], l[4], l[5])

        def vect_only(l):
            return l[2]

        def categorical_only(l):
            return l[3]

        # gen_poss,
        # this_turn_poss,
        # midpoint,
        # acts,
        # num_moves,
        # predicted_reward,
        labels = [l if l is not None else o for l, o in zip(labels, outputs)]
        scalar_x = scalar_only(outputs)
        scalar_y = scalar_only(labels)
        scalar_loss = sum(
            [torch.nn.functional.mse_loss(x, y) for x, y in zip(scalar_x, scalar_y)]
        )
        vect_x = vect_only(outputs)
        vect_y = vect_only(labels)
        vect_loss = torch.nn.functional.cosine_similarity(vect_x, vect_y)
        cat_x = categorical_only(outputs)
        cat_y = categorical_only(labels)
        cat_loss = torch.nn.functional.cross_entropy(cat_x, cat_y)
        return sum([scalar_loss, vect_loss, cat_loss])
