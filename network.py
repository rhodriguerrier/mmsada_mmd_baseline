import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 8)

    def forward(self, x):
        x = self.fc1(x)
        intermediate_feature = F.relu(x)
        output = F.softmax(self.fc2(intermediate_feature), dim=0)
        return intermediate_feature, output

class IntermediateNet(nn.Module):
    def __init__(self):
        super(IntermediateNet, self).__init__()
        self.fc1 = nn.Linear(1024, 1024)

    def forward(self, x):
        x = self.fc1(x)
        return F.relu(x)
