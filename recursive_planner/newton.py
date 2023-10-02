import torch
from agent import Brain
from torch.autograd.functional import jacobian
from torch.func import jacrev, vmap


def get_action(net: Brain, obs, goal):
    grad = torch.autograd.grad(net(obs, goal).action_optim_loss())
    return grad


# works batched
def optimize_subplan(
    net: Brain,
    obs: torch.Tensor,
    goal: torch.Tensor,
    noise: torch.Tensor,
    steps: int = 100,
):
    net = net.eval()
    midpoint_jacobian = vmap(
        jacrev(lambda x, y, z: net.unbatched_forward(x, y, z).subplan_optim_loss(), 2)
    )
    for _ in range(steps):
        grad = midpoint_jacobian(obs, goal, noise)
        grad = grad.squeeze(1)
        grad = grad * -1  # go up
        noise = noise - grad
    return noise


frame_1 = torch.randn(10, 224, 240, 3)
frame_2 = torch.randn(10, 224, 240, 3)
batch = frame_1.shape[0]
noise = torch.randn(batch, 16)
brain = Brain()

grad = optimize_subplan(brain, frame_1, frame_2, noise)
