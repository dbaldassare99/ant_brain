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


def test_brain_1_batch():
    frame_1 = torch.randn(1, 224, 240, 3)
    frame_2 = torch.randn(1, 224, 240, 3)
    noise = torch.randn(1, 16)
    brain = Brain().eval()  # normalize fails if not eval because of batchnorm
    out = brain(frame_1, frame_2, noise)
    assert out.vects.shape == torch.Size([1, 3, 16])
    assert out.gen_poss.shape == torch.Size([1, 1])
    assert out.this_turn_poss.shape == torch.Size([1, 1])
    assert out.midpoint.shape == torch.Size([1, 16])
    assert out.predicted_reward.shape == torch.Size([1, 1])
    assert out.num_moves.shape == torch.Size([1, 1])
    assert out.acts.shape == torch.Size([1, 10, 33])


def test_brain_unbatched():
    frame_1 = torch.randn(224, 240, 3)
    frame_2 = torch.randn(224, 240, 3)
    noise = torch.randn(16)
    brain = Brain().eval()  # normalize fails if not eval because of batchnorm
    out = brain.unbatched_forward(frame_1, frame_2, noise)
    assert out.vects.shape == torch.Size([3, 16])
    assert out.gen_poss.shape == torch.Size([1])
    assert out.this_turn_poss.shape == torch.Size([1])
    assert out.midpoint.shape == torch.Size([16])
    assert out.predicted_reward.shape == torch.Size([1])
    assert out.num_moves.shape == torch.Size([1])
    assert out.acts.shape == torch.Size([10, 33])


def test_brain_10_batch():
    frame_1 = torch.randn(10, 224, 240, 3)
    frame_2 = torch.randn(10, 224, 240, 3)
    noise = torch.randn(10, 16)
    brain = Brain()
    out = brain(frame_1, frame_2, noise)
    assert out.vects.shape == torch.Size([10, 3, 16])
    assert out.gen_poss.shape == torch.Size([10, 1])
    assert out.this_turn_poss.shape == torch.Size([10, 1])
    assert out.midpoint.shape == torch.Size([10, 16])
    assert out.predicted_reward.shape == torch.Size([10, 1])
    assert out.num_moves.shape == torch.Size([10, 1])
    assert out.acts.shape == torch.Size([10, 10, 33])


def test_get_actions():
    brain_out = State(None, None, None, None, None, None, None, None, None)
    brain_out.acts = torch.rand(5, 10, 33)
    brain_out.get_action_sequence()
