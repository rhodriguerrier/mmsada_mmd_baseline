import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from network import Net
from dataset import load_training_datasets, load_test_datasets
from mmd_loss import mix_rbf_mmd2
import pickle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


epochs = 100

d1_loader, d2_loader = load_training_datasets()
model = Net()
optim = torch.optim.Adam(model.parameters(), lr=0.0001)
for epoch in range(epochs):
    print(f"Epoch: {epoch}")
    add_mmd_loss = False
    if epoch > 35:
        add_mmd_loss = True
        optim.param_groups[0]['lr'] = 0.00005
    sum_loss = 0
    sum_mmd_loss = 0
    counter = 0
    for (d1_rgb_ft, d1_flow_ft, d1_labels), (d2_rgb_ft, d2_flow_ft, d2_labels) in zip(d1_loader, d2_loader):
        if d1_rgb_ft.shape != d2_rgb_ft.shape or d1_flow_ft.shape != d2_flow_ft.shape:
            continue
        ce_loss = nn.CrossEntropyLoss()
        new_d1_rgb_ft, new_d1_flow_ft, d1_output = model(torch.tensor(d1_rgb_ft).float(), torch.tensor(d1_flow_ft).float())
        new_d2_rgb_ft, new_d2_flow_ft, d2_output = model(torch.tensor(d2_rgb_ft).float(), torch.tensor(d2_flow_ft).float())
        d1_class_loss = ce_loss(d1_output, d1_labels.long())
        if add_mmd_loss:
            rgb_mmd_loss = mix_rbf_mmd2(
                new_d1_rgb_ft,
                new_d2_rgb_ft,
                gammas=[10.0,1.0,0.1,0.01,0.001]
            )
            flow_mmd_loss = mix_rbf_mmd2(
                new_d1_flow_ft,
                new_d2_flow_ft,
                gammas=[10.0,1.0,0.1,0.01,0.001]
            )
            loss = d1_class_loss + (1 * (rgb_mmd_loss + flow_mmd_loss))
            sum_mmd_loss += (rgb_mmd_loss + flow_mmd_loss)
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


sum_samples = 0
sum_correct = 0
d1_test_loader, d2_test_loader = load_test_datasets()
for (d1_rgb_test_ft, d1_flow_test_ft, d1_test_labels), (d2_rgb_test_ft, d2_flow_test_ft, d2_test_labels) in zip(d1_test_loader, d2_test_loader):
    new_d1_rgb_ft, new_d1_flow_ft, d1_output = model(torch.tensor(d1_rgb_test_ft).float(), torch.tensor(d1_flow_test_ft).float())
    new_d2_rgb_ft, new_d2_flow_ft, d2_output = model(torch.tensor(d2_rgb_test_ft).float(), torch.tensor(d2_flow_test_ft).float())
    d2_batch_results = torch.eq(torch.argmax(d2_output, dim=1), d2_test_labels.long()).long()
    num_correct = torch.sum(d2_batch_results)
    sum_samples += d2_batch_results.size(dim=0)
    sum_correct += num_correct.item()
print(sum_correct, sum_samples)
print(f"D2 Percentage correct = {(sum_correct / sum_samples) * 100}%")
