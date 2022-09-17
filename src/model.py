import torch.nn as nn


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()

        self.fc1 = nn.Linear(in_features=2, out_features=10, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=10, out_features=1, bias=True)

    def forward(self, x):
        """
        x: torch.Size([batch, 2]))
        """

        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = out.reshape(-1)
        return out


def get_model() -> MyNet:
    return MyNet()
