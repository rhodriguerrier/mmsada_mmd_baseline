from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import pandas as pd


class RgbFlowDataset(Dataset):
    def __init__(self, rgb_data_path, flow_data_path, labels_path):
        rgb_data_info = load_data(rgb_data_path)
        flow_data_info = load_data(flow_data_path)
        label_info = load_data(labels_path)
        self.rgb_data = reshape_data(rgb_data_info, "RGB")
        self.flow_data = reshape_data(flow_data_info, "Flow")
        ids = np.array(rgb_data_info['narration_ids'])
        ids_col = np.array([])
        for single_id in ids:
            id_num = int(single_id.split("_")[-1])
            verb_class = label_info.set_index('uid').loc[id_num, 'verb_class']
            ids = np.repeat(verb_class, 5)
            ids_col = np.concatenate((ids_col, ids), axis=0)
        self.labels = ids_col

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.rgb_data[index], self.flow_data[index], self.labels[index]


def load_data(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
        return data


def reshape_data(data, modality):
    shape = data['features'][modality].shape
    return data['features'][modality].reshape(shape[0]*shape[1], 1024)


def load_training_datasets(source_data_name, source_labels_name, target_data_name, target_labels_name, batch_size):
    d1_train = RgbFlowDataset(
        rgb_data_path=f"./pre_extracted_features/RGB/RGB/ek_i3d/{source_data_name}.pkl",
        flow_data_path=f"./pre_extracted_features/Flow/Flow/ek_i3d/{source_data_name}.pkl",
        labels_path=f"./label_lookup/{source_labels_name}.pkl"
    )
    d1_loader = DataLoader(d1_train, batch_size=64, shuffle=True)
    d2_train = RgbFlowDataset(
        rgb_data_path=f"./pre_extracted_features/RGB/RGB/ek_i3d/{target_data_name}.pkl",
        flow_data_path=f"./pre_extracted_features/Flow/Flow/ek_i3d/{target_data_name}.pkl",
        labels_path=f"./label_lookup/{target_labels_name}.pkl"
    )
    d2_loader = DataLoader(d2_train, batch_size=64, shuffle=True)
    return d1_loader, d2_loader


def load_test_datasets(source_data_name, source_labels_name, target_data_name, target_labels_name, batch_size):
    d1_train = RgbFlowDataset(
        rgb_data_path=f"./pre_extracted_features/RGB/RGB/ek_i3d/{source_data_name}.pkl",
        flow_data_path=f"./pre_extracted_features/Flow/Flow/ek_i3d/{source_data_name}.pkl",
        labels_path=f"./label_lookup/{source_labels_name}.pkl"
    )
    d1_loader = DataLoader(d1_train, batch_size=batch_size, shuffle=True)
    d2_train = RgbFlowDataset(
        rgb_data_path=f"./pre_extracted_features/RGB/RGB/ek_i3d/{target_data_name}.pkl",
        flow_data_path=f"./pre_extracted_features/Flow/Flow/ek_i3d/{target_data_name}.pkl",
        labels_path=f"./label_lookup/{target_labels_name}.pkl"
    )
    d2_loader = DataLoader(d2_train, batch_size=batch_size, shuffle=True)
    return d1_loader, d2_loader
