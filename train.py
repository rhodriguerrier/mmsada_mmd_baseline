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


#wandb.init(
#    project="mmsada_mmd_baseline",
#    name="run-7-ss",
#    config={
#        "initial_lr": 0.0001,
#        "secondary_lr": 0.00008,
#        "self_supervision": True,
#        "lambda_c": 5,
#        "epochs": 100,
#        "batch_size": 128,
#        "feature_dims": "1024 -> 1024 -> 512"
#    }
#)

class Model:
	def __init__(
			self,
			epochs,
			batch_size,
			initial_lr,
			secondary_lr
	):
		self.epochs = epochs
		self.batch_size = batch_size
		self.initial_lr = initial_lr
		self.secondary_lr = secondary_lr
		self.model = Net()
		self.source_train_loader, self.target_train_loader = load_training_datasets(
			"D1-D1_train",
			"D1_train",
			"D2-D2_train",
			"D2_train",
			self.batch_size
		)
		self.source_test_loader, self.target_test_loader = load_test_datasets(
			"D1-D1_test",
			"D1_test",
			"D2-D2_test",
			"D2_test",
			self.batch_size
		)
		self.optim = torch.optim.Adam(self.model.parameters(), lr=self.initial_lr)
		self.ce_loss = nn.CrossEntropyLoss()

	def train_model(self):
		for epoch in range(self.epochs):
			print(f"Epoch: {epoch}")
			add_mmd_loss = False
			if epoch > 35:
				add_mmd_loss = True
				self.optim.param_groups[0]['lr'] = self.secondary_lr
			sum_loss = 0
			sum_mmd_loss = 0
			sum_ss_loss = 0
			counter = 0
			for (d1_rgb_ft, d1_flow_ft, d1_labels), (d2_rgb_ft, d2_flow_ft, d2_labels) in zip(self.source_train_loader, self.target_train_loader):
				if d1_rgb_ft.shape != d2_rgb_ft.shape or d1_flow_ft.shape != d2_flow_ft.shape:
					continue
				new_d1_rgb_ft, new_d1_flow_ft, d1_output, d1_ss_output = self.model(torch.tensor(d1_rgb_ft).float(), torch.tensor(d1_flow_ft).float(), True)
				new_d2_rgb_ft, new_d2_flow_ft, d2_output, d2_ss_output = self.model(torch.tensor(d2_rgb_ft).float(), torch.tensor(d2_flow_ft).float(), True)
				d1_class_loss = self.ce_loss(d1_output, d1_labels.long())
				d1_ss_loss = self.ce_loss(d1_ss_output, torch.full((d1_ss_output.size()[0],), 1))
				d2_ss_loss = self.ce_loss(d2_ss_output, torch.full((d2_ss_output.size()[0],), 1))
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
					loss = d1_class_loss + (5 * (d1_ss_loss + d2_ss_loss)) + (1 * (rgb_mmd_loss + flow_mmd_loss))
					sum_ss_loss += (5*(d1_ss_loss + d2_ss_loss))
					sum_mmd_loss += (rgb_mmd_loss + flow_mmd_loss)
					sum_loss += loss
				else:
					loss = d1_class_loss + (5 * (d1_ss_loss + d2_ss_loss))
					sum_ss_loss += (d1_ss_loss + d2_ss_loss)
					sum_loss += loss
				counter += 1
				self.optim.zero_grad()
				loss.backward()
				self.optim.step()
			if add_mmd_loss:
				print(f"Loss = {sum_loss / counter}, SS Loss = {sum_ss_loss}, MMD Loss = {sum_mmd_loss / counter}")
				#wandb.log({"Total Loss": (sum_loss / counter)})
				#wandb.log({"Self-Supervision Loss": (sum_ss_loss / counter)})
				#wandb.log({"MMD Loss": (sum_mmd_loss / counter)})
			else:
				print(f"Loss = {sum_loss / counter}, SS Loss = {sum_ss_loss / counter}")
				#wandb.log({"Self-Supervision Loss": (sum_ss_loss / counter)})
				#wandb.log({"Total Loss": (sum_loss / counter)})

	def test(self):
		rgb_features = torch.tensor([])
		flow_features = torch.tensor([])
		rgb_domain_labels = np.array([])
		flow_domain_labels = np.array([])
		class_labels = np.array([])
		sum_samples = 0
		sum_correct = 0
		for (d1_rgb_test_ft, d1_flow_test_ft, d1_test_labels), (d2_rgb_test_ft, d2_flow_test_ft, d2_test_labels) in zip(self.source_test_loader, self.target_test_loader):
			new_d1_rgb_ft, new_d1_flow_ft, d1_output, d1_ss_output = self.model(torch.tensor(d1_rgb_test_ft).float(), torch.tensor(d1_flow_test_ft).float(), False)
			new_d2_rgb_ft, new_d2_flow_ft, d2_output, d2_ss_output = self.model(torch.tensor(d2_rgb_test_ft).float(), torch.tensor(d2_flow_test_ft).float(), False)

			d1_rgb_domain_labels = np.full(new_d1_rgb_ft.size()[0], 1)
			d2_rgb_domain_labels = np.full(new_d2_rgb_ft.size()[0], 2)
			d1_flow_domain_labels = np.full(new_d1_flow_ft.size()[0], 1)
			d2_flow_domain_labels = np.full(new_d2_flow_ft.size()[0], 2)

			rgb_features = torch.cat((rgb_features, new_d1_rgb_ft, new_d2_rgb_ft), 0)
			rgb_domain_labels = np.concatenate((rgb_domain_labels, d1_rgb_domain_labels, d2_rgb_domain_labels))
			class_labels = np.concatenate((class_labels, d1_test_labels, d2_test_labels))
			flow_features = torch.cat((flow_features, new_d1_flow_ft, new_d2_flow_ft), 0)
			flow_domain_labels = np.concatenate((flow_domain_labels, d1_flow_domain_labels, d2_flow_domain_labels))

			d2_batch_results = torch.eq(torch.argmax(d2_output, dim=1), d2_test_labels.long()).long()
			num_correct = torch.sum(d2_batch_results)
			sum_samples += d2_batch_results.size(dim=0)
			sum_correct += num_correct.item()
		print(sum_correct, sum_samples)
		print(f"D2 Percentage correct = {(sum_correct / sum_samples) * 100}%")
		return rgb_features, rgb_domain_labels, flow_features, flow_domain_labels, class_labels


def plot_data(
		rgb_ft,
		flow_ft,
		rgb_domain_labels,
		flow_domain_labels,
		class_labels,
		test_run_stage
):
	low_dim_rgb = TSNE(
	n_components=2,
		init='random'
	).fit_transform(rgb_ft)
	low_dim_flow = TSNE(
		n_components=2,
		init='random'
	).fit_transform(flow_ft)
	fig, axs = plt.subplots(2, 2)
	fig.suptitle(f"tSNE plot of features {test_run_stage} model trained")
	axs[0, 0].scatter(low_dim_rgb[:,0], low_dim_rgb[:,1], 1, c=rgb_domain_labels)
	axs[0, 0].set_title("RGB by Domain")
	axs[0, 1].scatter(low_dim_flow[:,0], low_dim_flow[:,1], 1, c=flow_domain_labels)
	axs[0, 1].set_title("Flow by Domain")
	axs[1, 0].scatter(low_dim_rgb[:,0], low_dim_rgb[:,1], 1, c=class_labels)
	axs[1, 0].set_title("RGB by Class")
	axs[1, 1].scatter(low_dim_flow[:,0], low_dim_flow[:,1], 1, c=class_labels)
	axs[1, 1].set_title("Flow by Class")
	fig.savefig(f"tsne_features_{test_run_stage}.png")


if __name__ == "__main__":
	model = Model(
		epochs=100,
		batch_size=128,
		initial_lr=0.0001,
		secondary_lr=0.00008
	)
	rgb_ft_before, rgb_domain_labels_before, flow_ft_before, flow_domain_labels_before, class_labels_before = model.test()
	plot_data(
		rgb_ft_before.detach().numpy(),
		flow_ft_before.detach().numpy(),
		rgb_domain_labels_before,
		flow_domain_labels_before,
		class_labels_before, 
		"before"
	)
	model.train_model()
	rgb_ft_after, rgb_domain_labels_after, flow_ft_after, flow_domain_labels_after, class_labels_after = model.test()
	plot_data(
		rgb_ft_after.detach().numpy(),
		flow_ft_after.detach().numpy(),
		rgb_domain_labels_after,
		flow_domain_labels_after,
		class_labels_after,
		"after"
	)	
