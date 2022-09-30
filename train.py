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
import wandb


wandb.init(project="mmsada_mmd_baseline", name="run-5", config={"initial_lr": 0.0001, "secondary_lr": 0.00008, "tertiary_lr": 0.00005, "epochs": 100, "batch_size": 128})

epochs = 100

d1_loader, d2_loader = load_training_datasets()
model = Net()
optim = torch.optim.Adam(model.parameters(), lr=0.0001)
for epoch in range(epochs):
    print(f"Epoch: {epoch}")
    add_mmd_loss = False
    if epoch > 35:
        add_mmd_loss = True
        optim.param_groups[0]['lr'] = 0.00008
    if epoch > 80:
        optim.param_groups[0]['lr'] = 0.00005
    sum_loss = 0
    sum_mmd_loss = 0
    counter = 0
    for (d1_rgb_ft, d1_flow_ft, d1_labels), (d2_rgb_ft, d2_flow_ft, d2_labels) in zip(d1_loader, d2_loader):
        if d1_rgb_ft.shape != d2_rgb_ft.shape or d1_flow_ft.shape != d2_flow_ft.shape:
            continue
        ce_loss = nn.CrossEntropyLoss()
        new_d1_rgb_ft, new_d1_flow_ft, d1_output, d1_ss_output = model(torch.tensor(d1_rgb_ft).float(), torch.tensor(d1_flow_ft).float())
        new_d2_rgb_ft, new_d2_flow_ft, d2_output, d2_ss_output = model(torch.tensor(d2_rgb_ft).float(), torch.tensor(d2_flow_ft).float())
        d1_class_loss = ce_loss(d1_output, d1_labels.long())
        if add_mmd_loss:
            rgb_mmd_loss = mix_rbf_mmd2(
                new_d1_rgb_ft,
                new_d2_rgb_ft,
                gammas=[(2.0 ** gamma) * 9.7 for gamma in np.arange(-8.0, 8.0, 2.0 ** 0.5)]
            )
            flow_mmd_loss = mix_rbf_mmd2(
                new_d1_flow_ft,
                new_d2_flow_ft,
                gammas=[(2.0 ** gamma) * 9.7 for gamma in np.arange(-8.0, 8.0, 2.0 ** 0.5)]
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
        wandb.log({"Total Loss": (sum_loss / counter)})
        wandb.log({"MMD Loss": (sum_mmd_loss / counter)})
    else:
        print(f"Loss = {sum_loss / counter}")
        wandb.log({"Total Loss": (sum_loss / counter)})


rgb_features = torch.tensor([])
flow_features = torch.tensor([])
rgb_domain_labels = np.array([])
flow_domain_labels = np.array([])
sum_samples = 0
sum_correct = 0
d1_test_loader, d2_test_loader = load_test_datasets()
for (d1_rgb_test_ft, d1_flow_test_ft, d1_test_labels), (d2_rgb_test_ft, d2_flow_test_ft, d2_test_labels) in zip(d1_test_loader, d2_test_loader):
    new_d1_rgb_ft, new_d1_flow_ft, d1_output, d1_ss_output = model(torch.tensor(d1_rgb_test_ft).float(), torch.tensor(d1_flow_test_ft).float())
    new_d2_rgb_ft, new_d2_flow_ft, d2_output, d2_ss_output = model(torch.tensor(d2_rgb_test_ft).float(), torch.tensor(d2_flow_test_ft).float())

    d1_rgb_domain_labels = np.full(new_d1_rgb_ft.size()[0], 1)
    d2_rgb_domain_labels = np.full(new_d2_rgb_ft.size()[0], 2)
    d1_flow_domain_labels = np.full(new_d1_flow_ft.size()[0], 1)
    d2_flow_domain_labels = np.full(new_d2_flow_ft.size()[0], 2)

    rgb_features = torch.cat((rgb_features, new_d1_rgb_ft, new_d2_rgb_ft), 0)
    rgb_domain_labels = np.concatenate((rgb_domain_labels, d1_rgb_domain_labels, d2_rgb_domain_labels))
    flow_features = torch.cat((flow_features, new_d1_flow_ft, new_d2_flow_ft), 0)
    flow_domain_labels = np.concatenate((flow_domain_labels, d1_flow_domain_labels, d2_flow_domain_labels))

    d2_batch_results = torch.eq(torch.argmax(d2_output, dim=1), d2_test_labels.long()).long()
    num_correct = torch.sum(d2_batch_results)
    sum_samples += d2_batch_results.size(dim=0)
    sum_correct += num_correct.item()
print(sum_correct, sum_samples)
print(f"D2 Percentage correct = {(sum_correct / sum_samples) * 100}%")


all_rgb_ft = rgb_features.detach().numpy()
all_flow_ft = flow_features.detach().numpy()
low_dim_rgb = TSNE(
    n_components=2,
    init='random'
).fit_transform(all_rgb_ft)
low_dim_flow = TSNE(
    n_components=2,
    init='random'
).fit_transform(all_flow_ft)
fig, (rgb_ax, flow_ax) = plt.subplots(1, 2)
fig.suptitle("tSNE Plot of Features After Trained Model")
rgb_ax.scatter(low_dim_rgb[:,0], low_dim_rgb[:,1], 1, c=rgb_domain_labels)
rgb_ax.set_title("RGB")
flow_ax.scatter(low_dim_flow[:,0], low_dim_flow[:,1], 1, c=flow_domain_labels)
flow_ax.set_title("Flow")
fig.savefig("tsne_features_after.png")
