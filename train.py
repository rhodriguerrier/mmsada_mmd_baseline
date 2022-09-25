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


epochs = 10
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
optim = torch.optim.Adam(model.parameters(), lr=0.0001)
for epoch in range(epochs):
	print(f"Epoch: {epoch}")
	sum_loss = 0
	sum_mmd_loss = 0
	counter = 0
	for (d1_features, d1_labels), (d2_features, d2_labels) in zip(d1_loader, d2_loader):
		if d1_features.shape != d2_features.shape:
			continue
		ce_loss = nn.CrossEntropyLoss()
		new_d1_features, d1_output = model(torch.tensor(d1_features).float())
		new_d2_features, d2_output = model(torch.tensor(d2_features).float())
		mmd_loss = mix_rbf_mmd2(
			new_d1_features,
			new_d2_features,
			gammas=[10.0,1.0,0.1,0.01,0.001]
		)
		d1_class_loss = ce_loss(d1_output, d1_labels.long())
		d2_class_loss = ce_loss(d2_output, d2_labels.long())
		counter += 1
		sum_mmd_loss += mmd_loss
		loss = d1_class_loss + d2_class_loss + (0.2 * mmd_loss)
		sum_loss += loss
		if counter % 10 == 0:
			print(f"Loss = {sum_loss / counter}, MMD Loss = {sum_mmd_loss / counter}")
		optim.zero_grad()
		loss.backward()
		optim.step()
