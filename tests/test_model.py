import torch
from model import MyNet


def test_mynet():
    net = MyNet()
    x = torch.zeros(2)
    y = net(x)
    assert y.shape == torch.Size([1])
