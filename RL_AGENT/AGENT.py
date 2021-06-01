import torch as T
import torch.nn as nn

s = [51.1, 0.85, 14.07, 9.849]

class DQN_PRED(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Sequential(
                                nn.Linear(4, 20, bias=True),
                                nn.BatchNorm2d(10),
                                nn.ReLU(inplace=True),
                                nn.Linear(20, 10, bias=True),
                                nn.Softmax(dim=1)
                                )
    def forward(self, x):
        inp = T.tensor(x)
        inp = inp.unsqueeze(dim=0)
        inp = inp.unsqueeze(dim=1)
        inp = self.lin(inp)
        return inp
