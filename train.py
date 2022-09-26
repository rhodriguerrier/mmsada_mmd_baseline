import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from network import Net
from dataset import CustomDataset
from mmd_loss import mix_rbf_mmd2
import pickle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


epochs = 100
d1_train = CustomDataset(
    data_path="./pre_extracted_features/RGB/RGB/ek_i3d/D1-D1_train.pkl",
    labels_path="./label_lookup/D1_train.pkl",
    modality="RGB"
)
d1_loader = DataLoader(d1_train, batch_size=64, shuffle=True)

d2_train = CustomDataset(
    data_path="./pre_extracted_features/RGB/RGB/ek_i3d/D2-D2_train.pkl",
    labels_path="./label_lookup/D2_train.pkl",
    modality="RGB"
)
d2_loader = DataLoader(d2_train, batch_size=64, shuffle=True)

model = Net()
optim = torch.optim.Adam(model.parameters(), lr=0.00001)
for epoch in range(epochs):
    print(f"Epoch: {epoch}")
    add_mmd_loss = False
    if epoch > 35:
        add_mmd_loss = True
        #optim.param_groups[0]['lr'] = 0.000001
    sum_loss = 0
    sum_mmd_loss = 0
    counter = 0
    for (d1_features, d1_labels), (d2_features, d2_labels) in zip(d1_loader, d2_loader):
        if d1_features.shape != d2_features.shape:
            continue
        ce_loss = nn.CrossEntropyLoss()
        new_d1_features, d1_output = model(torch.tensor(d1_features).float())
        new_d2_features, d2_output = model(torch.tensor(d2_features).float())
        d1_class_loss = ce_loss(d1_output, d1_labels.long())
        if add_mmd_loss:
            mmd_loss = mix_rbf_mmd2(
                new_d1_features,
                new_d2_features,
                gammas=[10.0,1.0,0.1,0.01,0.001]
            )
            loss = d1_class_loss + (1 * mmd_loss)
            sum_mmd_loss += mmd_loss
            sum_loss += loss
        else:
            loss = d1_class_loss
            sum_loss += loss
        counter += 1
        optim.zero_grad()
        loss.backward()
        optim.step()
    if add_mmd_loss:
        print(f"Loss = {sum_loss / counter}, MMD Loss = {sum_mmd_loss / counter}")
    else:
        print(f"Loss = {sum_loss / counter}")

d1_test = CustomDataset(
    data_path="./pre_extracted_features/RGB/RGB/ek_i3d/D1-D1_test.pkl",
    labels_path="./label_lookup/D1_test.pkl",
    modality="RGB"
)
d1_test_loader = DataLoader(d1_test, batch_size=64, shuffle=True)
d2_test = CustomDataset(
    data_path="./pre_extracted_features/Flow/Flow/ek_i3d/D2-D2_test.pkl",
    labels_path="./label_lookup/D2_test.pkl",
    modality="Flow"
)
d2_test_loader = DataLoader(d2_test, batch_size=64, shuffle=True)
sum_samples = 0
sum_correct = 0
for (d1_test_features, d1_test_labels), (d2_test_features, d2_test_labels) in zip(d1_test_loader, d2_test_loader):
    new_d1_features, d1_output = model(torch.tensor(d1_test_features).float())
    new_d2_features, d2_output = model(torch.tensor(d2_test_features).float())
    d1_batch_results = torch.eq(torch.argmax(d1_output, dim=1), d1_test_labels.long()).long()
    d2_batch_results = torch.eq(torch.argmax(d2_output, dim=1), d2_test_labels.long()).long()
    num_correct = torch.sum(d1_batch_results) + torch.sum(d2_batch_results)
    sum_samples += d1_batch_results.size(dim=0) + d2_batch_results.size(dim=0)
    sum_correct += num_correct.item()
print(sum_correct, sum_samples)
print(f"Percentage Correct = {(sum_correct / sum_samples) * 100}")
