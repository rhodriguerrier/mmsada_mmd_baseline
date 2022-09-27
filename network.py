import torch
import torch.nn as nn
import torch.nn.functional as F

#class Net(nn.Module):
#    def __init__(self):
#        super(Net, self).__init__()
#        self.fc1 = nn.Linear(1024, 1024)
#        self.fc2 = nn.Linear(1024, 8)
#
#    def forward(self, x):
#        x = self.fc1(x)
#        intermediate_feature = F.relu(x)
#        output = F.softmax(self.fc2(intermediate_feature), dim=1)
#        return intermediate_feature, output


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1_rgb = nn.Linear(1024, 1024)
        self.fc1_flow = nn.Linear(1024, 1024)

        self.fc2_rgb = nn.Linear(1024, 8)
        self.fc2_flow = nn.Linear(1024, 8)

    def forward(self, x_rgb, x_flow):
        x_rgb = self.fc1_rgb(x_rgb)
        x_flow = self.fc1_flow(x_flow)
        mid_rgb_features = F.relu(x_rgb)
        mid_flow_features = F.relu(x_flow)
        rgb_class_logits = self.fc2_rgb(mid_rgb_features)
        flow_class_logits = self.fc2_flow(mid_flow_features)
        output = F.softmax((rgb_class_logits + flow_class_logits), dim=1)
        return mid_rgb_features, mid_flow_features, output
