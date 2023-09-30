from agent import Brain
import torch


def test_brain():
    brain = Brain()
    frame_1 = torch.randn(10, 224, 240, 3)
    frame_2 = torch.randn(10, 224, 240, 3)
    noise = torch.randn(10, 16)
    brain = Brain()
    out = brain(frame_1, frame_2, noise)
    assert (
        str(out)
        == """
        BrainOutput
        vects: torch.Size([10, 3, 16])
        gen_poss: torch.Size([10, 1])
        this_turn_poss: torch.Size([10, 1])
        mid: torch.Size([10, 16])
        predicted reward: torch.Size([10, 1])
        predicted moves: torch.Size([10, 1])
        acts: torch.Size([10, 10, 33])
        """
    )
