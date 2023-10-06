from nets import Brain, BrainOut
from buffer import State
import torch


def test_brain_batchless():
    obs = torch.randn(224, 240, 3)
    goal = torch.randn(16)
    noise = torch.randn(16)
    brain = Brain().eval()  # normalize fails if not eval because of batchnorm
    outs = BrainOut(brain(obs, goal, noise))
    assert outs.gen_poss.shape == torch.Size([1])
    assert outs.poss_this_turn.shape == torch.Size([1])
    assert outs.midpoint.shape == torch.Size([16])
    assert outs.num_moves.shape == torch.Size([1])
    assert outs.acts.shape == torch.Size([37, 10])
    assert outs.rew.shape == torch.Size([1])


def test_brain_1_batch():
    obs = torch.randn(1, 224, 240, 3)
    goal = torch.randn(1, 16)
    noise = torch.randn(1, 16)
    brain = Brain().eval()  # normalize fails if not eval because of batchnorm
    outs = BrainOut(brain(obs, goal, noise))
    assert outs.gen_poss.shape == torch.Size([1, 1])
    assert outs.poss_this_turn.shape == torch.Size([1, 1])
    assert outs.midpoint.shape == torch.Size([1, 16])
    assert outs.num_moves.shape == torch.Size([1, 1])
    assert outs.acts.shape == torch.Size([1, 37, 10])
    assert outs.rew.shape == torch.Size([1, 1])


def test_brain_10_batch():
    obs = torch.randn(10, 224, 240, 3)
    goal = torch.randn(10, 16)
    noise = torch.randn(10, 16)
    brain = Brain()
    outs = BrainOut(brain(obs, goal, noise))
    assert outs.gen_poss.shape == torch.Size([10, 1])
    assert outs.poss_this_turn.shape == torch.Size([10, 1])
    assert outs.midpoint.shape == torch.Size([10, 16])
    assert outs.num_moves.shape == torch.Size([10, 1])
    assert outs.acts.shape == torch.Size([10, 37, 10])
    assert outs.rew.shape == torch.Size([10, 1])


def test_optimize_subplan():
    obs = torch.randn(224, 240, 3)
    goal = torch.randn(16)
    noise = torch.randn(16)
    state = State(obs=obs, goal=goal, noise=noise, gen_poss=0.0)
    brain = Brain()
    before, after = state.optimize_subplan(brain)
    assert before < after


def test_optimize_plan():
    obs = torch.randn(224, 240, 3)
    goal = torch.randn(16)
    noise = torch.randn(16)
    state = State(obs=obs, goal=goal, noise=noise, gen_poss=0.0)
    brain = Brain()
    before, after = state.optimize_plan(brain)
    assert before < after


def test_get_action_sequence():
    brain = Brain()
    state = State()
    state.obs = torch.randn(224, 240, 3)
    state.update(brain)
    action_sequence = state.get_action_sequence()
    assert action_sequence[0].shape == torch.Size([10])


if __name__ == "__main__":
    test_brain_1_batch()
    test_brain_10_batch()
    test_optimize_subplan()
    test_optimize_plan()
