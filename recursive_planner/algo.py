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
import pytorch_lightning as pl


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
        # self.brain.load_state_dict(
        #     torch.load(
        #         "/home/cibo/ant_brain/recursive_planner/model_checkpoints/brain.pt"
        #     )
        # )
        self.mem_notes = MemoryBuffer()
        self.buffer = ExperienceBuffer()

    def play(self):
        self.random_bootstrap()
        obs, info = self.env.reset()
        state = State(obs=obs)

        while True:
            self.brain.eval()
            print("\nmaking new plan")
            state.optimize_plan(self.brain)
            print("\nacting")
            memory = self.recursive_actor(state)
            state.obs = memory.last_obs
            # self.train()

    def random_bootstrap(self) -> None:
        _ = self.env.reset()
        random_play_len = 500
        max_sample_length = 200
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
        depth: int = 0,
    ) -> Memory:
        depth += 1
        arg_state = copy.copy(state)  # to see how close we got!
        state.optimize_subplan(self.brain)
        if depth > 3:
            state.poss_this_turn = 1
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
            memory_part_1 = self.recursive_actor(state_part_1, depth)
            # second half
            state_part_2 = copy.copy(state)
            state_part_2.obs = memory_part_1.last_obs
            memory_part_2 = self.recursive_actor(state_part_2, depth)
            # third half (lol) (really 2nd half again)
            state_part_3 = copy.copy(state)
            state_part_3.obs = memory_part_2.last_obs
            memory_part_3 = self.recursive_actor(state_part_3, depth)

            # get info abt the plan
            check_state = copy.copy(state)
            check_state.obs = memory_part_3.last_obs
            check_state.optimize_subplan(self.brain)
            print("expected moves left in plan:", check_state.num_moves)
            plan_record = Memory(state)
            plan_record.add_plan([memory_part_1, memory_part_2, memory_part_3])
            plan_record.gen_poss = (
                1
                if check_state.num_moves < 3
                # or arg_state.rew - plan_record.predicted_reward <= 3
                else 0
            )
            self.mem_notes.add(plan_record)
            return plan_record

    def make_trainer_loop(self, net, buffer):
        generator1 = torch.Generator().manual_seed(42)
        exp_dataset = buffer.to_Dataset(net)
        train, val = torch.utils.data.random_split(
            exp_dataset, [0.7, 0.3], generator=generator1
        )
        train = torch.utils.data.DataLoader(train, shuffle=True, num_workers=8)
        val = torch.utils.data.DataLoader(val, shuffle=False, num_workers=8)
        trainer = pl.Trainer(limit_train_batches=100)
        trainer.fit(net, train, val)

    def train(self, train_vision: bool = False):
        if train_vision:
            self.make_trainer_loop(self.vision_trainer, self.buffer)
            # generator1 = torch.Generator().manual_seed(42)
            # exp_dataset = self.buffer.to_Dataset(self.brain)
            # train, val = torch.utils.data.random_split(
            #     exp_dataset, [0.7, 0.3], generator=generator1
            # )
            # train = torch.utils.data.DataLoader(train, shuffle=True, num_workers=8)
            # val = torch.utils.data.DataLoader(val, shuffle=True, num_workers=8)
            # trainer = pl.Trainer(limit_train_batches=100)
            # trainer.fit(self.vision_trainer, train, val)

            # self.vision_trainer.train()
            # self.vision_trainer.requires_grad_(True)
            # optimizer = torch.optim.Adam(self.vision_trainer.parameters(), lr=1e-3)
            # pbar = trange(400, desc="vision_train")
            # for i in pbar:
            #     optimizer.zero_grad()
            #     ins, labels = self.buffer.vision_examples(64, i == -1)
            #     outputs = self.vision_trainer(*ins)
            #     loss, (act, rew) = self.vision_trainer.loss(outputs, labels)
            #     loss.backward()
            #     optimizer.step()
            #     pbar.set_postfix_str(f"losses: act: {act:.2f} rew: {rew:.2f}")
            # torch.save(
            #     self.vision_trainer.state_dict(),
            #     "/home/cibo/ant_brain/recursive_planner/model_checkpoints/vision_trainer.pt",
            # )
        self.make_trainer_loop(self.brain, self.mem_notes)
        # generator1 = torch.Generator().manual_seed(42)
        # mem_dataset = self.mem_notes.to_Dataset(self.brain)
        # train, val = torch.utils.data.random_split(
        #     mem_dataset, [0.7, 0.3], generator=generator1
        # )
        # train = torch.utils.data.DataLoader(train, shuffle=True, num_workers=8)
        # val = torch.utils.data.DataLoader(val, shuffle=True, num_workers=8)
        # trainer = pl.Trainer(limit_train_batches=100)
        # trainer.fit(self.brain, train, val)

    def loss_fn(self, outputs: BrainOut, labels: BrainOut):
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
