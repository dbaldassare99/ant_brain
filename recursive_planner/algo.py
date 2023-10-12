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
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class RP:
    def __init__(self, env: TorchGym) -> None:
        self.env = env
        self.vision = Vision()
        self.vision_trainer = VisionTrainer(self.vision).float()
        self.vision_path = (
            "/home/cibo/ant_brain/recursive_planner/model_checkpoints/vision_trainer.pt"
        )
        self.vision_trainer.load_state_dict(torch.load(self.vision_path))
        self.brain = Brain(self.vision)
        self.brain_path = (
            "/home/cibo/ant_brain/recursive_planner/model_checkpoints/brain.pt"
        )
        # self.brain.load_state_dict(torch.load(self.brain_path))
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
        random_play_len = 50
        max_sample_length = 40
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

    def make_trainer_loop(self, net, buffer, path):
        generator1 = torch.Generator().manual_seed(42)
        dataset = buffer.to_Dataset(net)
        train = torch.utils.data.DataLoader(
            dataset, shuffle=True, num_workers=8, batch_size=3
        )
        # train, val = torch.utils.data.random_split(
        #     exp_dataset, [0.7, 0.3], generator=generator1
        # )
        # train = torch.utils.data.DataLoader(train, shuffle=True, num_workers=8)
        # val = torch.utils.data.DataLoader(val, shuffle=False, num_workers=8)
        trainer = pl.Trainer(
            limit_train_batches=100,
            # callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
            max_epochs=2,
        )
        # trainer.fit(net, train, val)
        trainer.fit(net, train)
        torch.save(net.state_dict(), path)

    def train(self, train_vision: bool = False):
        if train_vision:
            self.make_trainer_loop(self.vision_trainer, self.buffer)
        self.make_trainer_loop(self.brain, self.mem_notes, self.brain_path)
