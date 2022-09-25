from dataset import CustomerDataset
from mmd_loss import mix_rbf_mmd2
import pickle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


epochs = 30
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

model = IntermediateNet()
optim = torch.optim.SGD(model.parameters(), lr=0.02)
for epoch in range(epochs):
    print(f"Epoch: {epoch}")
    sum_loss = 0
    counter = 0
    for (d1_features, d1_labels), (d2_features, d2_labels) in zip(d1_loader, d2_loader):
        if d1_features.shape != d2_features.shape:
            continue
        new_d1_features = model(torch.tensor(d1_features).float())
        new_d2_features = model(torch.tensor(d2_features).float())
        loss = mix_rbf_mmd2(
            new_d1_features,
            new_d2_features,
            gammas=[10.0,1.0,0.1,0.01,0.001]
        )
        counter += 1
        sum_loss += loss
        if counter % 10 == 0:
            print(f"Loss: {sum_loss / counter}")
        optim.zero_grad()
        loss.backward()
        optim.step()
