from agent import Brain
from buffer import State
import torch

# self.vects = vects
# self.gen_poss = gen_poss
# self.this_turn_poss = this_turn_poss
# self.midpoint = midpoint
# self.predicted_reward = predicted_reward
# self.predicted_moves = predicted_moves
# self.acts = acts


def test_brain_0_batch():
    frame_1 = torch.randn(224, 240, 3)
    frame_2 = torch.randn(224, 240, 3)
    noise = torch.randn(16)
    brain = Brain().eval()  # normalize fails if not eval because of batchnorm
    (
        generally_possibe,
        possibe_this_turn,
        midpoint,
        acts,
        num_moves,
        predicted_reward,
    ) = brain(frame_1, frame_2, noise)
    assert generally_possibe.shape == torch.Size([1])
    assert possibe_this_turn.shape == torch.Size([1])
    assert midpoint.shape == torch.Size([16])
    assert num_moves.shape == torch.Size([1])
    assert acts.shape == torch.Size([10, 33])
    assert predicted_reward.shape == torch.Size([1])


def test_brain_1_batch():
    frame_1 = torch.randn(1, 224, 240, 3)
    frame_2 = torch.randn(1, 224, 240, 3)
    noise = torch.randn(1, 16)
    brain = Brain().eval()  # normalize fails if not eval because of batchnorm
    (
        generally_possibe,
        possibe_this_turn,
        midpoint,
        acts,
        num_moves,
        predicted_reward,
    ) = brain(frame_1, frame_2, noise)
    assert generally_possibe.shape == torch.Size([1, 1])
    assert possibe_this_turn.shape == torch.Size([1, 1])
    assert midpoint.shape == torch.Size([1, 16])
    assert num_moves.shape == torch.Size([1, 1])
    assert acts.shape == torch.Size([1, 10, 33])
    assert predicted_reward.shape == torch.Size([1, 1])


def test_brain_10_batch():
    frame_1 = torch.randn(10, 224, 240, 3)
    frame_2 = torch.randn(10, 224, 240, 3)
    noise = torch.randn(10, 16)
    brain = Brain()
    (
        generally_possibe,
        possibe_this_turn,
        midpoint,
        acts,
        num_moves,
        predicted_reward,
    ) = brain(frame_1, frame_2, noise)
    assert generally_possibe.shape == torch.Size([10, 1])
    assert possibe_this_turn.shape == torch.Size([10, 1])
    assert midpoint.shape == torch.Size([10, 16])
    assert num_moves.shape == torch.Size([10, 1])
    assert acts.shape == torch.Size([10, 10, 33])
    assert predicted_reward.shape == torch.Size([10, 1])


def test_optimize_subplan():
    frame_1 = torch.randn(224, 240, 3)
    frame_2 = torch.randn(224, 240, 3)
    noise = torch.randn(16)
    state = State(obs=frame_1, goal=frame_2, noise=noise, gen_poss=0.0)
    brain = Brain()
    before, after = state.optimize_subplan(brain)
    assert before < after


def test_optimize_plan():
    frame_1 = torch.randn(224, 240, 3)
    frame_2 = torch.randn(224, 240, 3)
    noise = torch.randn(16)
    state = State(obs=frame_1, goal=frame_2, noise=noise, gen_poss=0.0)
    brain = Brain()
    before, after = state.optimize_plan(brain)
    assert before < after


if __name__ == "__main__":
    test_brain_1_batch()
    test_brain_10_batch()
    print("subplan")
    test_optimize_subplan()
    print("plan")
    test_optimize_plan()
