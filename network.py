import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #Intermediate features
        self.fc1_rgb = nn.Linear(1024, 1024)
        self.fc1_flow = nn.Linear(1024, 1024)

        # Self-supervision
        self.fc2_a_1 = nn.Linear(2048, 100)
        self.fc2_a_2 = nn.Linear(100, 2)

        # Classification heads
        self.fc2_b_rgb = nn.Linear(1024, 8)
        self.fc2_b_flow = nn.Linear(1024, 8)

    def forward(self, x_rgb, x_flow):
        # Extract intermediate features
        x_rgb = self.fc1_rgb(x_rgb)
        x_flow = self.fc1_flow(x_flow)
        mid_rgb_features = F.relu(x_rgb)
        mid_flow_features = F.relu(x_flow)

        # Send intermediate features to self-supervision
        concat_features = torch.cat((mid_rgb_features, mid_flow_features), -1)
        mid_self_sup = F.relu(self.fc2_a_1(concat_features))
        ss_logits = self.fc2_a_2(mid_self_sup)

        # Send intermediate features to classification head
        rgb_class_logits = self.fc2_b_rgb(mid_rgb_features)
        flow_class_logits = self.fc2_b_flow(mid_flow_features)
        class_logits = (rgb_class_logits + flow_class_logits)
        return mid_rgb_features, mid_flow_features, class_logits, ss_logits
