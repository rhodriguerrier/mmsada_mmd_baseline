import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from network import Net, ConcatFeaturesNet, DomainDiscriminator
from dataset import load_datasets
from mmd_loss import mix_rbf_mmd2
import pickle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import wandb
import argparse


#wandb.init(
#    project="mmsada_mmd_baseline",
#    name="d1-to-d2-adversarial-17",
#    config={
#        "initial_lr": 0.0001,
#        "self_supervision": True,
#        "lambda_c": 5,
#        "epochs": 150,
#        "batch_size": 128,
#        "batch_norm": True,
#        "dropout": 0.5,
#        "weight_decay": 0.0000001,
#        "feature_in_format": "concatenate",
#        "feature_dims": "5120 -> 1024 -> 512",
#        "adversarial": True
#    }
#)

net_selector = {
    "separate": Net,
    "concatenate": ConcatFeaturesNet,
    "average": Net
}

class Model:
    def __init__(
            self,
            epochs,
            batch_size,
            initial_lr,
            src_dom_name,
            trg_dom_name,
            action_seg_format
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.initial_lr = initial_lr
        self.model = net_selector[action_seg_format]()
        self.rgb_discrim_model = DomainDiscriminator()
        self.rgb_discrim_optim = torch.optim.SGD(
            self.rgb_discrim_model.parameters(),
            lr=self.initial_lr,
            momentum=0.9,
            weight_decay=0.0000001
        )
        self.flow_discrim_model = DomainDiscriminator()
        self.flow_discrim_optim = torch.optim.SGD(
            self.flow_discrim_model.parameters(),
            lr=self.initial_lr,
            momentum=0.9,
            weight_decay=0.0000001
        )
        self.source_train_loader, self.target_train_loader = load_datasets(
            f"{src_dom_name}-{src_dom_name}_train.pkl",
            f"{src_dom_name}_train.pkl",
            f"{src_dom_name}-{trg_dom_name}_train.hkl",
            f"{trg_dom_name}_train.pkl",
            self.batch_size,
            source_hickle=False,
            target_hickle=True,
            action_seg_format=action_seg_format
        )
        self.source_test_loader, self.target_test_loader = load_datasets(
            f"{src_dom_name}-{src_dom_name}_test.pkl",
            f"{src_dom_name}_test.pkl",
            f"{src_dom_name}-{trg_dom_name}_test.pkl",
            f"{trg_dom_name}_test.pkl",
            self.batch_size,
            source_hickle=False,
            target_hickle=False,
            action_seg_format=action_seg_format
        )
        self.optim = torch.optim.SGD(self.model.parameters(), lr=self.initial_lr, momentum=0.9, weight_decay=0.0000001)
        self.ce_loss = nn.CrossEntropyLoss()


    def train_adversarial(self):
        self.model.train()
        self.rgb_discrim_model.train()
        self.flow_discrim_model.train()
        for epoch in range(self.epochs):
            print(f"Epoch: {epoch}")
            rgb_ft_after, rgb_domain_labels_after, flow_ft_after, flow_domain_labels_after, class_labels_after = self.test()
            if epoch == 0:
                plot_data(
                    rgb_ft_after.detach().numpy(),
                    flow_ft_after.detach().numpy(),
                    rgb_domain_labels_after,
                    flow_domain_labels_after,
                    class_labels_after,
                    "before model trained",
                    "tsne_features_before"
                )
            elif epoch == 50:
                plot_data(
                    rgb_ft_after.detach().numpy(),
                    flow_ft_after.detach().numpy(),
                    rgb_domain_labels_after,
                    flow_domain_labels_after,
                    class_labels_after,
                    "half way through Adversarial",
                    "tsne_features_mid_adv"
                )
            self.model.train()
            sum_loss = 0
            sum_rgb_discrim_loss = 0
            sum_flow_discrim_loss = 0
            sum_rgb_ft_loss = 0
            sum_flow_ft_loss = 0
            sum_ss_loss = 0
            sum_src_cls_loss = 0
            counter = 0
            for (src_rgb_ft, src_flow_ft, src_labels), (tgt_rgb_ft, tgt_flow_ft, tgt_labels) in zip(self.source_train_loader, self.target_train_loader):
                new_src_rgb_ft, new_src_flow_ft, src_output, src_ss_output = self.model(torch.tensor(src_rgb_ft).float(), torch.tensor(src_flow_ft).float(), False)
                new_tgt_rgb_ft, new_tgt_flow_ft, tgt_output, tgt_ss_output = self.model(torch.tensor(tgt_rgb_ft).float(), torch.tensor(tgt_flow_ft).float(), True)
                rgb_ft_loss = 0
                flow_ft_loss = 0
                # discriminator loss here
                self.rgb_discrim_optim.zero_grad()
                self.flow_discrim_optim.zero_grad()

                rgb_concat_ft = torch.cat((new_src_rgb_ft, new_tgt_rgb_ft), dim=0)
                rgb_domain_predictions = self.rgb_discrim_model(rgb_concat_ft)
                rgb_src_domain_labels = torch.ones(new_src_rgb_ft.size(0)).long()
                rgb_tgt_domain_labels = torch.zeros(new_tgt_rgb_ft.size(0)).long()
                rgb_discrim_loss = self.ce_loss(rgb_domain_predictions, torch.cat((rgb_src_domain_labels, rgb_tgt_domain_labels)))
                sum_rgb_discrim_loss += rgb_discrim_loss
                rgb_discrim_loss.backward()
                self.rgb_discrim_optim.step()

                flow_concat_ft = torch.cat((new_src_flow_ft, new_tgt_flow_ft), dim=0)
                flow_domain_predictions = self.flow_discrim_model(flow_concat_ft)
                flow_src_domain_labels = torch.ones(new_src_flow_ft.size(0)).long()
                flow_tgt_domain_labels = torch.zeros(new_tgt_flow_ft.size(0)).long()
                flow_discrim_loss = self.ce_loss(flow_domain_predictions, torch.cat((flow_src_domain_labels, flow_tgt_domain_labels)))
                sum_flow_discrim_loss += flow_discrim_loss
                flow_discrim_loss.backward()
                self.flow_discrim_optim.step()

                # feature loss here
                self.rgb_discrim_optim.zero_grad()
                self.flow_discrim_optim.zero_grad()
                new_src_rgb_ft, new_src_flow_ft, src_output, src_ss_output = self.model(torch.tensor(src_rgb_ft).float(), torch.tensor(src_flow_ft).float(), False)
                new_tgt_rgb_ft, new_tgt_flow_ft, tgt_output, tgt_ss_output = self.model(torch.tensor(tgt_rgb_ft).float(), torch.tensor(tgt_flow_ft).float(), True)

                rgb_concat_ft = torch.cat((new_src_rgb_ft, new_tgt_rgb_ft), dim=0)
                rgb_domain_predictions = self.rgb_discrim_model(rgb_concat_ft)
                rgb_src_domain_labels = torch.zeros(new_src_rgb_ft.size(0)).long()
                rgb_tgt_domain_labels = torch.ones(new_tgt_rgb_ft.size(0)).long()
                rgb_ft_loss = self.ce_loss(rgb_domain_predictions, torch.cat((rgb_src_domain_labels, rgb_tgt_domain_labels)))
                sum_rgb_ft_loss += rgb_ft_loss

                flow_concat_ft = torch.cat((new_src_flow_ft, new_tgt_flow_ft), dim=0)
                flow_domain_predictions = self.flow_discrim_model(flow_concat_ft)
                flow_src_domain_labels = torch.zeros(new_src_flow_ft.size(0)).long()
                flow_tgt_domain_labels = torch.ones(new_tgt_flow_ft.size(0)).long()
                flow_ft_loss = self.ce_loss(flow_domain_predictions, torch.cat((flow_src_domain_labels, flow_tgt_domain_labels)))
                sum_flow_ft_loss += flow_ft_loss

                src_class_loss = self.ce_loss(src_output, src_labels.long())
                src_ss_loss = self.ce_loss(src_ss_output, torch.full((src_ss_output.size()[0],), 1))
                tgt_ss_loss = self.ce_loss(tgt_ss_output, torch.full((tgt_ss_output.size()[0],), 1))
                loss = src_class_loss + (5 * (src_ss_loss + tgt_ss_loss)) + (1 * (rgb_ft_loss + flow_ft_loss))
                sum_ss_loss += (src_ss_loss + tgt_ss_loss)
                sum_loss += loss
                sum_src_cls_loss += src_class_loss
                counter += 1
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
            print(f"Loss = {sum_loss / counter}, SS Loss = {sum_ss_loss}")
            print(f"RGB Discriminator Loss = {sum_rgb_discrim_loss / counter}, RGB Feature Loss = {sum_rgb_ft_loss / counter}")
            print(f"Flow Discriminator Loss = {sum_flow_discrim_loss / counter}, Flow Feature Loss = {sum_flow_ft_loss / counter}")
            #wandb.log({"Total Loss": (sum_loss / counter)})
            #wandb.log({"Source Classification Loss": (sum_src_cls_loss / counter)})
            #wandb.log({"Self-Supervision Loss": (sum_ss_loss / counter)})
            #wandb.log({"RGB Discriminator Loss": (sum_rgb_discrim_loss / counter)})
            #wandb.log({"RGB Feature Loss": (sum_rgb_ft_loss / counter)})
            #wandb.log({"Flow Discriminator Loss": (sum_flow_discrim_loss / counter)})
            #wandb.log({"Flow Feature Loss": (sum_flow_ft_loss / counter)})
            
 
    def test(self):
        self.model.eval()
        rgb_features = torch.tensor([])
        flow_features = torch.tensor([])
        rgb_domain_labels = np.array([])
        flow_domain_labels = np.array([])
        class_labels = np.array([])
        sum_samples = 0
        sum_correct = 0
        for (d1_rgb_test_ft, d1_flow_test_ft, d1_test_labels) in self.source_test_loader:
            new_d1_rgb_ft, new_d1_flow_ft, d1_output, d1_ss_output = self.model(torch.tensor(d1_rgb_test_ft).float(), torch.tensor(d1_flow_test_ft).float(), False)

            d1_rgb_domain_labels = np.full(new_d1_rgb_ft.size()[0], 1)
            d1_flow_domain_labels = np.full(new_d1_flow_ft.size()[0], 1)

            rgb_features = torch.cat((rgb_features, new_d1_rgb_ft), 0)
            rgb_domain_labels = np.concatenate((rgb_domain_labels, d1_rgb_domain_labels))
            class_labels = np.concatenate((class_labels, d1_test_labels))
            flow_features = torch.cat((flow_features, new_d1_flow_ft), 0)
            flow_domain_labels = np.concatenate((flow_domain_labels, d1_flow_domain_labels))
        for (d2_rgb_test_ft, d2_flow_test_ft, d2_test_labels) in self.target_test_loader:
            new_d2_rgb_ft, new_d2_flow_ft, d2_output, d2_ss_output = self.model(torch.tensor(d2_rgb_test_ft).float(), torch.tensor(d2_flow_test_ft).float(), True)

            d2_rgb_domain_labels = np.full(new_d2_rgb_ft.size()[0], 2)
            d2_flow_domain_labels = np.full(new_d2_flow_ft.size()[0], 2)

            rgb_features = torch.cat((rgb_features, new_d2_rgb_ft), 0)
            rgb_domain_labels = np.concatenate((rgb_domain_labels, d2_rgb_domain_labels))
            class_labels = np.concatenate((class_labels, d2_test_labels))
            flow_features = torch.cat((flow_features, new_d2_flow_ft), 0)
            flow_domain_labels = np.concatenate((flow_domain_labels, d2_flow_domain_labels))

            d2_batch_results = torch.eq(torch.argmax(d2_output, dim=1), d2_test_labels.long()).long()
            num_correct = torch.sum(d2_batch_results)
            sum_samples += d2_batch_results.size(dim=0)
            sum_correct += num_correct.item()
        print(sum_correct, sum_samples)
        print(f"Target Percentage correct = {(sum_correct / sum_samples) * 100}%")
        #wandb.log({"Target Test Accuracy": ((sum_correct / sum_samples) * 100)})
        return rgb_features, rgb_domain_labels, flow_features, flow_domain_labels, class_labels


def plot_data(
        rgb_ft,
        flow_ft,
        rgb_domain_labels,
        flow_domain_labels,
        class_labels,
        test_run_stage,
        file_name
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
    fig.suptitle(f"tSNE plot of features {test_run_stage}")
    axs[0, 0].scatter(low_dim_rgb[:,0], low_dim_rgb[:,1], 1, c=rgb_domain_labels)
    axs[0, 0].set_title("RGB by Domain")
    axs[0, 1].scatter(low_dim_flow[:,0], low_dim_flow[:,1], 1, c=flow_domain_labels)
    axs[0, 1].set_title("Flow by Domain")
    axs[1, 0].scatter(low_dim_rgb[:,0], low_dim_rgb[:,1], 1, c=class_labels)
    axs[1, 0].set_title("RGB by Class")
    axs[1, 1].scatter(low_dim_flow[:,0], low_dim_flow[:,1], 1, c=class_labels)
    axs[1, 1].set_title("Flow by Class")
    fig.savefig(f"{file_name}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MMD Baseline App")
    parser.add_argument("--src_dom_name", action="store", dest="src_dom_name", default="D1")
    parser.add_argument("--trg_dom_name", action="store", dest="trg_dom_name", default="D2")
    parser.add_argument("--batch_size", action="store", dest="batch_size", default="128")
    parser.add_argument("--epochs", action="store", dest="epochs", default="150")
    parser.add_argument("--initial_lr", action="store", dest="initial_lr", default="0.0001")
    parser.add_argument("--action_seg_format", action="store", dest="action_seg_format", default="concatenate")
    args = parser.parse_args()
 
    model = Model(
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        initial_lr=float(args.initial_lr),
        src_dom_name=args.src_dom_name,
        trg_dom_name=args.trg_dom_name,
        action_seg_format=args.action_seg_format
    )
    model.train_adversarial()
    rgb_ft_after, rgb_domain_labels_after, flow_ft_after, flow_domain_labels_after, class_labels_after = model.test()
    plot_data(
        rgb_ft_after.detach().numpy(),
        flow_ft_after.detach().numpy(),
        rgb_domain_labels_after,
        flow_domain_labels_after,
        class_labels_after,
        "after model trained with adversarial",
        "tsne_features_after_adversarial"
    )
