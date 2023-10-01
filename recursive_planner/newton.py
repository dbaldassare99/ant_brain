import torch
from agent import Brain
from torch.autograd.functional import jacobian
from torch.func import jacrev, vmap


def get_action(net: Brain, obs, goal):
    grad = torch.autograd.grad(net(obs, goal).action_optim_loss())
    return grad


def get_midpoint(net: Brain, obs, goal):
    net = net.eval()
    batch = obs.shape[0]
    noise = torch.randn(batch, 16)
    midpoint_jacobian = vmap(
        jacrev(lambda x, y, z: net.unbatched_forward(x, y, z).midpoint_optim_loss(), 2)
    )
    for _ in range(10):
        print(net(obs, goal, noise).midpoint_optim_loss())
        print(obs.shape, goal.shape, noise.shape)
        grad = midpoint_jacobian(obs, goal, noise)
        grad = grad.unsqueeze(1)
        noise = noise - grad


frame_1 = torch.randn(10, 224, 240, 3)
frame_2 = torch.randn(10, 224, 240, 3)
brain = Brain()

grad = get_midpoint(brain, frame_1, frame_2)
