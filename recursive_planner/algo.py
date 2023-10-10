# Get Goal
# Recursive bad boy:
from nets import VisionTrainer, BrainOut, Vision, Brain
from tqdm import trange
import numpy as np
import torch
from buffer import (
    MemoryBuffer,
    State,
    StateQueue,
    Memory,
    Timestep,
    SMA,
    TorchGym,
    ExperienceBuffer,
)
import copy


class RP:
    def __init__(self, env: TorchGym) -> None:
        self.env = env
        self.vision = Vision()
        self.vision_trainer = VisionTrainer(self.vision).float()
        self.vision_trainer.load_state_dict(
            torch.load(
                "/home/cibo/ant_brain/recursive_planner/model_checkpoints/vision_trainer.pt"
            )
        )
        self.brain = Brain(self.vision)
        self.brain.load_state_dict(
            torch.load(
                "/home/cibo/ant_brain/recursive_planner/model_checkpoints/brain.pt"
            )
        )
        self.mem_notes = MemoryBuffer()
        self.buffer = ExperienceBuffer()

    def play(self):
        # self.random_bootstrap()
        obs, info = self.env.reset()
        state = State(obs=obs)

        while True:
            state.optimize_plan(self.brain)
            memory = self.recursive_actor(state)
            state.obs = memory.last_obs

    def random_bootstrap(self) -> None:
        _ = self.env.reset()
        random_play_len = 500
        max_sample_length = 100
        assert max_sample_length < random_play_len
        for i in range(random_play_len):
            if i % 20 == 0:
                action = self.env.rand_act()
            self.buffer.add(Timestep(*self.env.step(action), action))

        for _ in range(1_000):
            start = np.random.randint(0, random_play_len - max_sample_length)
            end = start + min(
                np.random.randint(1, max_sample_length),
                np.random.randint(1, max_sample_length),
                np.random.randint(1, max_sample_length),
                np.random.randint(1, max_sample_length),
            )

            mem = Memory(State()).rand_start(self.buffer, start, end)
            self.mem_notes.add(mem)

        self.train()
        self.mem_notes = MemoryBuffer()

    def recursive_actor(
        self,
        state: State,
    ) -> Memory:
        self.brain.eval()
        arg_state = copy.copy(state)  # to see how close we got!
        state.optimize_subplan(self.brain)
        if state.poss_this_turn >= 0.5:
            action_sequence = state.get_action_sequence()
            print(action_sequence)
            queue = StateQueue()
            for action in action_sequence[0]:  # not parallel
                queue(self.env.step(action))
            # get info abt the action
            state.optimize_subplan(self.brain)
            action_record = Memory(state)
            action_record.add_action(queue, action_sequence)
            self.mem_notes.add(action_record)
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
            check_state.optimize_subplan(self.brain)
            plan_record = Memory(state)
            plan_record.add_plan([memory_part_1, memory_part_2, memory_part_3])
            plan_record.gen_poss = (
                1
                if check_state.num_moves < 3
                or arg_state.rew - plan_record.predicted_reward <= 3
                else 0
            )
            self.mem_notes.add(plan_record)
            return plan_record

    def train(self, train_vision: bool = False):
        if train_vision:
            self.vision_trainer.train()
            self.vision_trainer.requires_grad_(True)
            optimizer = torch.optim.Adam(self.vision_trainer.parameters(), lr=1e-3)
            pbar = trange(400, desc="vision_train")
            for i in pbar:
                optimizer.zero_grad()
                ins, labels = self.buffer.vision_examples(64, i == -1)
                outputs = self.vision_trainer(*ins)
                loss, (act, rew) = self.vision_trainer.loss(outputs, labels)
                loss.backward()
                optimizer.step()
                pbar.set_postfix_str(f"losses: act: {act:.2f} rew: {rew:.2f}")
            torch.save(
                self.vision_trainer.state_dict(),
                "/home/cibo/ant_brain/recursive_planner/model_checkpoints/vision_trainer.pt",
            )

        self.brain.train()
        self.brain.vision.requires_grad_(False)
        loss_sma = SMA(10)
        optimizer = torch.optim.Adam(self.brain.parameters(), lr=1e-3)
        total_examples = 30_000
        batch_size = 64
        total_steps = total_examples // batch_size
        pbar = trange(total_steps)
        for i in pbar:
            optimizer.zero_grad()
            ins, labels, learn_from = self.mem_notes.sample_preprocess_and_batch(
                self.brain, batch_size, i == 1 and batch_size <= 10
            )
            outputs = BrainOut(self.brain(*ins))
            loss, (scalar, vect, cat) = self.loss_fn(outputs, labels, learn_from)
            loss.backward()
            optimizer.step()
            loss_sma.add(loss.item())
            pbar.set_postfix_str(f"losses: {scalar:.2f} {vect:.2f} {cat:.2f}")
        torch.save(
            self.brain.state_dict(),
            "/home/cibo/ant_brain/recursive_planner/model_checkpoints/brain.pt",
        )

    def loss_fn(self, outputs: BrainOut, labels: BrainOut, learn_from: list):
        def scalar_only(l: BrainOut):
            return l.gen_poss, l.poss_this_turn, l.num_moves, l.rew

        def list_mean(l: list[torch.Tensor]) -> torch.Tensor:
            return sum(l) / len(l)

        def print_scalar(o, l):
            print(torch.stack([o, l], dim=1).squeeze())

        # print(learn_from)
        # print(labels.acts)
        # for i, replace in enumerate(learn_from):
        #     labels.acts[i:...] = outputs.acts[i:...] if replace else labels.acts[i:...]
        # print(labels.acts)
        # assert 2 == 3

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

        # print_scalar(outputs.gen_poss, labels.gen_poss)
        # print_scalar(outputs.poss_this_turn, labels.poss_this_turn)
        # print_scalar(outputs.rew, labels.rew)
        # print_scalar(outputs.num_moves, labels.num_moves)

        return list_mean([scalar_loss, vect_loss, cat_loss]), (
            scalar_loss.item(),
            vect_loss.item(),
            cat_loss.item(),
        )
