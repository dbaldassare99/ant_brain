# Get Goal
# Recursive bad boy:
from nets import Brain, BrainOut
from tqdm import trange
import numpy as np
import torch
from buffer import ExperienceBuffer, State, StateQueue, Memory, Timestep, SMA, TorchGym
import copy


class RP:
    def __init__(self, env: TorchGym) -> None:
        self.env = env
        self.net = Brain().float()
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
        random_play_len = 800
        max_sample_length = 100
        assert max_sample_length < random_play_len
        for i in range(random_play_len):
            if i % 20 == 0:
                action = self.env.rand_act()
            buffer.append(Timestep(*self.env.step(action), action))

        for _ in range(5):
            start = np.random.randint(0, random_play_len - max_sample_length)
            # end = start + np.random.randint(1, max_sample_length)
            end = start + min(
                np.random.randint(1, max_sample_length),
                np.random.randint(1, max_sample_length),
                np.random.randint(1, max_sample_length),
                np.random.randint(1, max_sample_length),
                np.random.randint(1, max_sample_length),
            )
            mem = Memory(State()).rand_start(buffer, start, end)
            self.notes.add(mem)

        self.train()
        self.notes = ExperienceBuffer()

    def recursive_actor(
        self,
        state: State,
    ) -> Memory:
        self.net.eval()
        arg_state = copy.copy(state)  # to see how close we got!
        state.optimize_subplan(self.net)
        if state.poss_this_turn >= 0.5:
            action_sequence = state.get_action_sequence()
            print(action_sequence)
            queue = StateQueue()
            for action in action_sequence[0]:  # not parallel
                queue(self.env.step(action))
            # get info abt the action
            state.optimize_subplan(self.net)
            action_record = Memory(state)
            action_record.add_action(queue, action_sequence)
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
            plan_record = Memory(state)
            plan_record.add_plan([memory_part_1, memory_part_2, memory_part_3])
            plan_record.gen_poss = (
                1
                if check_state.num_moves < 3
                or arg_state.rew - plan_record.predicted_reward <= 3
                else 0
            )
            self.notes.add(plan_record)
            return plan_record

    def train(self):
        self.net.train()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-4)
        pbar = trange(40_000)
        loss_sma = SMA(10)
        for i in pbar:
            optimizer.zero_grad()
            ins, labels = self.notes.sample_preprocess_and_batch(self.net, 3)
            outputs = BrainOut(self.net(*ins))
            loss, (scalar, vect, cat) = self.loss_fn(outputs, labels)
            print(loss)
            loss.backward()
            optimizer.step()
            loss_sma.add(loss.item())
            pbar.set_postfix_str(f"losses: {scalar:.2f} {vect:.2f} {cat:.2f}")

    def loss_fn(self, outputs: BrainOut, labels: BrainOut):
        def scalar_only(l: BrainOut):
            return l.gen_poss, l.poss_this_turn, l.num_moves, l.rew

        def vect_only(l: BrainOut):
            return l.midpoint

        def categorical_only(l: BrainOut):
            return l.acts

        def list_mean(l: list[torch.Tensor]) -> torch.Tensor:
            return sum(l) / len(l)

        # labels.replace_if_none(outputs)
        labels.gen_poss = outputs.gen_poss
        labels.poss_this_turn = outputs.poss_this_turn
        labels.midpoint = outputs.midpoint
        labels.acts = outputs.acts
        labels.num_moves = outputs.num_moves
        # labels.rew = outputs.rew

        scalar_x = scalar_only(outputs)
        scalar_y = scalar_only(labels)
        scalar_losses = [
            torch.nn.functional.mse_loss(x, y).mean()
            for x, y in zip(scalar_x, scalar_y)
        ]
        scalar_loss = list_mean(scalar_losses)
        vect_x = vect_only(outputs)
        vect_y = vect_only(labels)
        vect_loss = torch.nn.functional.cosine_similarity(vect_x, vect_y).mean()
        cat_x = categorical_only(outputs)
        cat_y = categorical_only(labels)
        # print(torch.argmax(cat_x, dim=1))
        # print(torch.argmax(cat_y, dim=1))
        # print(torch.argmax(vect_x, dim=1))
        # print(torch.argmax(vect_y, dim=1))
        # print(vect_x)
        # print(vect_y)
        # for xs, ys in zip(scalar_x, scalar_y):
        #     example = torch.stack([xs, ys], dim=1).squeeze()
        #     print(example)
        cat_loss = torch.nn.functional.cross_entropy(cat_x, cat_y).mean()

        print(torch.stack([outputs.rew, labels.rew], dim=1).squeeze())
        rew_loss = torch.nn.functional.l1_loss(outputs.rew, labels.rew)
        return rew_loss, (
            scalar_loss.item(),
            vect_loss.item(),
            cat_loss.item(),
        )
        # return list_mean([scalar_loss, vect_loss, cat_loss]).float(), (
        #     scalar_loss.item(),
        #     vect_loss.item(),
        #     cat_loss.item(),
        # )
