import torch
from agent import Brain


def get_acts(net: Brain, obs, goal):
    grad = torch.autograd.grad(net(obs, goal).action_optim_loss())
