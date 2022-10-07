import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #Intermediate features
        self.fc1_rgb_1 = nn.Linear(1024, 1024)
        self.fc1_rgb_2 = nn.Linear(1024, 512)
        self.fc1_flow_1 = nn.Linear(1024, 1024)
        self.fc1_flow_2 = nn.Linear(1024, 512)

        # Self-supervision
        self.fc2_a_1 = nn.Linear(1024, 100)
        self.fc2_a_2 = nn.Linear(100, 2)

        # Classification heads
        self.fc2_b_rgb = nn.Linear(512, 8)
        self.fc2_b_flow = nn.Linear(512, 8)

    def forward(self, x_rgb, x_flow, is_training):
        # Extract intermediate features
        x_rgb = F.relu(self.fc1_rgb_1(x_rgb))
        x_flow = F.relu(self.fc1_flow_1(x_flow))
        mid_rgb_features = F.relu(self.fc1_rgb_2(x_rgb))
        mid_flow_features = F.relu(self.fc1_flow_2(x_flow))

        # Send intermediate features to self-supervision
        concat_features = torch.cat((mid_rgb_features, mid_flow_features), -1)
        mid_self_sup = F.relu(self.fc2_a_1(concat_features))
        ss_logits = self.fc2_a_2(mid_self_sup)

        # Send intermediate features to classification head
        rgb_class_logits = self.fc2_b_rgb(mid_rgb_features)
        flow_class_logits = self.fc2_b_flow(mid_flow_features)
        class_logits = (rgb_class_logits + flow_class_logits)
        return F.dropout(mid_rgb_features, p=0.5, training=is_training), F.dropout(mid_flow_features, p=0.5, training=is_training), class_logits, ss_logits


class ConcatFeaturesNet(nn.Module):
    def __init__(self):
        super(ConcatFeaturesNet, self).__init__()
        #Intermediate features
        self.fc1_rgb_1 = nn.Linear(5120, 1024)
        self.fc1_rgb_2 = nn.Linear(1024, 512)
        self.fc1_flow_1 = nn.Linear(5120, 1024)
        self.fc1_flow_2 = nn.Linear(1024, 512)

        # Self-supervision
        self.fc2_a_1 = nn.Linear(1024, 100)
        self.fc2_a_2 = nn.Linear(100, 2)

        # Classification heads
        self.fc2_b_rgb = nn.Linear(512, 8)
        self.fc2_b_flow = nn.Linear(512, 8)

		# Extras
        self.bn1_src_rgb = nn.BatchNorm1d(1024)
        self.bn1_src_flow = nn.BatchNorm1d(1024)
        self.bn1_trg_rgb = nn.BatchNorm1d(1024)
        self.bn1_trg_flow = nn.BatchNorm1d(1024)
        self.bn2_src_rgb = nn.BatchNorm1d(512)
        self.bn2_src_flow = nn.BatchNorm1d(512)
        self.bn2_trg_rgb = nn.BatchNorm1d(512)
        self.bn2_trg_flow = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x_rgb, x_flow, is_target, is_training):
        # Extract intermediate features
        if is_target:
            x_rgb = self.bn1_trg_rgb(F.relu(self.fc1_rgb_1(x_rgb)))
            x_flow = self.bn1_trg_flow(F.relu(self.fc1_flow_1(x_flow)))
            mid_rgb_features = self.bn2_trg_rgb(F.relu(self.fc1_rgb_2(x_rgb)))
            mid_flow_features = self.bn2_trg_flow(F.relu(self.fc1_flow_2(x_flow)))
        else:
            x_rgb = self.bn1_src_rgb(F.relu(self.fc1_rgb_1(x_rgb)))
            x_flow = self.bn1_src_flow(F.relu(self.fc1_flow_1(x_flow)))
            mid_rgb_features = self.bn2_src_rgb(F.relu(self.fc1_rgb_2(x_rgb)))
            mid_flow_features = self.bn2_src_flow(F.relu(self.fc1_flow_2(x_flow)))
 
        # Send intermediate features to self-supervision
        concat_features = torch.cat((mid_rgb_features, mid_flow_features), -1)
        mid_self_sup = F.relu(self.fc2_a_1(concat_features))
        ss_logits = self.fc2_a_2(mid_self_sup)

        # Send intermediate features to classification head
        rgb_class_logits = self.fc2_b_rgb(mid_rgb_features)
        flow_class_logits = self.fc2_b_flow(mid_flow_features)
        class_logits = (rgb_class_logits + flow_class_logits)
        return self.dropout(mid_rgb_features), self.dropout(mid_flow_features), class_logits, ss_logits
