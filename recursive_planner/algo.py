# Get Goal
# Recursive bad boy:
from nets import VisionTrainer, BrainOut, Vision, Brain
import numpy as np
import torch
from buffer import (
    MemoryBuffer,
    State,
    Memory,
    Timestep,
    TorchGym,
    ExperienceBuffer,
)
import copy
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class RP:
    def __init__(self, env: TorchGym) -> None:
        self.env = env
        self.vision = Vision()
        self.vision_trainer = VisionTrainer(self.vision)
        self.vision_path = (
            "/home/cibo/ant_brain/recursive_planner/model_checkpoints/vision_trainer.pt"
        )
        self.vision_trainer.load_state_dict(torch.load(self.vision_path))
        self.brain = Brain(self.vision)
        self.brain_path = (
            "/home/cibo/ant_brain/recursive_planner/model_checkpoints/brain.pt"
        )
        self.brain.load_state_dict(torch.load(self.brain_path))
        self.mem_notes = MemoryBuffer()
        self.buffer = ExperienceBuffer()

    def play(self):
        # self.random_bootstrap()
        obs, info = self.env.reset()
        state = State(obs=obs)
        count = 1
        while True:
            count += 1
            self.brain.eval()
            print("\nmaking new plan")
            state.optimize_plan(self.brain)
            print("\nacting")
            self.recursive_actor(state)
            state.obs = self.buffer[-1].obs
            if count % 20 == 0:
                self.train()

    def random_bootstrap(self) -> None:
        _ = self.env.reset()
        random_play_len = 5_000
        max_sample_length = 100
        assert max_sample_length < random_play_len
        for i in range(random_play_len):
            if i % 20 == 0:
                action = self.env.rand_act()
            self.buffer.add(Timestep(*self.env.step(action), action))

        for _ in range(5_000):
            start = np.random.randint(0, random_play_len - max_sample_length)
            end = start + min(
                np.random.randint(1, max_sample_length),
                np.random.randint(1, max_sample_length),
                np.random.randint(1, max_sample_length),
                np.random.randint(1, max_sample_length),
            )
            len = end - start
            gen_poss = 1 if len < 40 else 0
            poss_this_turn = 1 if len < 10 else 0
            mem = Memory(
                buffer=self.buffer,
                start=start,
                gen_poss=gen_poss,
                poss_this_turn=poss_this_turn,
                end=end,
            )
            self.mem_notes.add(mem)

        self.buffer = ExperienceBuffer()
        self.train()
        self.mem_notes = MemoryBuffer()

    def recursive_actor(
        self,
        state: State,
        depth: int = 0,
    ) -> Memory:
        depth += 1
        state.optimize_subplan(self.brain)
        start = len(self.buffer) - 1
        if depth > 2:  # short plans only
            state.poss_this_turn = 1
        acted = False
        if state.poss_this_turn >= 0.5:
            acted = True
            action_sequence = state.get_action_sequence()
            print(action_sequence[0])
            for action in action_sequence[0]:  # not parallel
                self.buffer.add(Timestep(*self.env.step(action), action))
        else:
            sub_state = copy.copy(state)
            sub_state.goal = state.midpoint
            self.recursive_actor(sub_state, depth)
            sub_state.goal = state.goal
            sub_state.obs = self.buffer[-1].obs
            self.recursive_actor(sub_state, depth)
            sub_state.obs = self.buffer[-1].obs
            self.recursive_actor(sub_state, depth)

        state.obs = self.buffer[-1].obs
        state.optimize_subplan(self.brain)
        print("expected moves left in plan:", f"{state.num_moves.item():.2f}")

        poss_this_turn = 1 if state.num_moves <= 3 else 0
        poss_this_turn = poss_this_turn if acted else None
        gen_poss = 1 if state.num_moves <= 5 else 0  # is this ever none? rew too?
        if gen_poss == 1:
            print("good plan saved")
        if poss_this_turn == 1:
            print("good action saved")
        mem = Memory(
            self.buffer,
            start=start,
            gen_poss=gen_poss,
            poss_this_turn=poss_this_turn,
        )
        self.mem_notes.add(mem)

    def make_trainer_loop(self, net, buffer, path):
        # generator1 = torch.Generator().manual_seed(42)
        dataset = buffer.to_Dataset(net)
        train = torch.utils.data.DataLoader(
            dataset, shuffle=True, num_workers=2, batch_size=16
        )
        # train, val = torch.utils.data.random_split(
        #     exp_dataset, [0.7, 0.3], generator=generator1
        # )
        # train = torch.utils.data.DataLoader(train, shuffle=True, num_workers=8)
        # val = torch.utils.data.DataLoader(val, shuffle=False, num_workers=8)
        trainer = pl.Trainer(
            limit_train_batches=100,
            # callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
            max_epochs=3,
            log_every_n_steps=2,
        )
        # trainer.fit(net, train, val)

        trainer.fit(net, train)
        torch.save(net.state_dict(), path)

    def train(self, train_vision: bool = False):
        if train_vision:
            self.make_trainer_loop(self.vision_trainer, self.buffer)
        self.make_trainer_loop(self.brain, self.mem_notes, self.brain_path)
