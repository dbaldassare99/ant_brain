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
        random_play_len = 500
        max_sample_length = 100
        num_frames = 3
        max_sample_length -= num_frames  # for rgb last frame
        assert max_sample_length < random_play_len
        for i in range(random_play_len):
            if i % 20 == 0:
                action = self.env.rand_act()
            buffer.append(Timestep(*self.env.step(action), action))

        for _ in range(1_000):
            start = np.random.randint(num_frames, random_play_len - max_sample_length)
            end = start + min(
                # end = start + np.random.randint(5, max_sample_length)
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
        optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        total_examples = 20_000
        batch_size = 64
        total_steps = total_examples // batch_size
        pbar = trange(total_steps)
        loss_sma = SMA(10)
        for i in pbar:
            self.net.vision.requires_grad_(
                i < int(total_steps * 0.9)
            )  # train vision at first
            optimizer.zero_grad()
            ins, labels, learn_from = self.notes.sample_preprocess_and_batch(
                self.net, batch_size, i == 1 and batch_size <= 10
            )
            outputs = BrainOut(self.net(*ins))
            loss, (scalar, vect, cat) = self.loss_fn(outputs, labels, learn_from)
            loss.backward()
            optimizer.step()
            loss_sma.add(loss.item())
            pbar.set_postfix_str(f"losses: {scalar:.2f} {vect:.2f} {cat:.2f}")

    def loss_fn(self, outputs: BrainOut, labels: BrainOut, learn_from: list):
        def scalar_only(l: BrainOut):
            return l.gen_poss, l.poss_this_turn, l.num_moves, l.rew

        def list_mean(l: list[torch.Tensor]) -> torch.Tensor:
            return sum(l) / len(l)

        print(learn_from)
        print(labels.acts)
        for i, replace in enumerate(learn_from):
            labels.acts[i:...] = outputs.acts[i:...] if replace else labels.acts[i:...]
        print(labels.acts)
        assert 2 == 3

        scalar_losses = [
            torch.nn.functional.mse_loss(x, y).mean()
            for x, y in zip(scalar_only(outputs), scalar_only(labels))
        ]
        scalar_loss = list_mean(scalar_losses)
        vect_loss = 1 - (
            torch.nn.functional.cosine_similarity(
                outputs.midpoint, labels.midpoint
            ).mean()
        )
        cat_loss = torch.nn.functional.cross_entropy(outputs.acts, labels.acts).mean()

        # print(torch.stack([outputs.gen_poss, labels.gen_poss], dim=1).squeeze())
        # print(
        #     torch.stack(
        #         [outputs.poss_this_turn, labels.poss_this_turn], dim=1
        #     ).squeeze()
        # )
        # print(torch.stack([outputs.rew, labels.rew], dim=1).squeeze())
        # print(torch.stack([outputs.num_moves, labels.num_moves], dim=1).squeeze())
        return list_mean([scalar_loss, vect_loss, cat_loss]), (
            scalar_loss.item(),
            vect_loss.item(),
            cat_loss.item(),
        )
