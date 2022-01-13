import torch
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot


class Network(nn.Module):

    def __init__(self, feature_dim, out_dim):
        super(Network, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.model = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# https://stackoverflow.com/questions/44130851/simple-lstm-in-pytorch-with-sequential-module
class ExtractLastCell(nn.Module):
    def forward(self,x):
        out , _ = x
        return out[:, -1, :]

class NetworkLSTM(nn.Module):

    def __init__(self, feature_dim, out_dim):
        super(NetworkLSTM, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.model = nn.Sequential(
            nn.LSTM(feature_dim, 32, batch_first=True),
            ExtractLastCell(),
            #nn.ReLU(),
            #nn.LSTM(128, 128),
            #nn.ReLU(),
            nn.Linear(32, out_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    x = torch.zeros(1, 5, dtype=torch.float, requires_grad=False)
    model = Network(feature_dim=5, out_dim=4)
    out = model(x)

    print(model)
    #make_dot(out, params=dict(list(model.named_parameters()))).view()