from torch.utils.data import Dataset, DataLoader
import pickle
import hickle as hkl
import numpy as np
import pandas as pd


class RgbFlowDataset(Dataset):
    def __init__(self, rgb_data_path, flow_data_path, labels_path, is_hickle):
        if is_hickle:
            rgb_data_info = load_hickle_data(rgb_data_path)
            flow_data_info = load_hickle_data(flow_data_path)
            ids = [x.decode("utf-8") for x in list(rgb_data_info["'narration_ids'"])]
            self.rgb_data = reshape_action_segs(rgb_data_info["'features'"]["'RGB'"][0])
            self.flow_data = reshape_action_segs(flow_data_info["'features'"]["'Flow'"][0])
        else:
            rgb_data_info = load_pickle_data(rgb_data_path)
            flow_data_info = load_pickle_data(flow_data_path)
            self.rgb_data = reshape_action_segs(rgb_data_info["features"]["RGB"])
            self.flow_data = reshape_action_segs(flow_data_info["features"]["Flow"])
            ids = np.array(rgb_data_info['narration_ids'])
        label_info = load_pickle_data(labels_path)
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


class RgbFlowConcatDataset(Dataset):
    def __init__(self, rgb_data_path, flow_data_path, labels_path, is_hickle):
        if is_hickle:
            rgb_data_info = load_hickle_data(rgb_data_path)
            flow_data_info = load_hickle_data(flow_data_path)
            ids = [x.decode("utf-8") for x in list(rgb_data_info["'narration_ids'"])]
            self.rgb_data = concat_action_segs(rgb_data_info["'features'"]["'RGB'"][0])
            self.flow_data = concat_action_segs(flow_data_info["'features'"]["'Flow'"][0])
        else:
            rgb_data_info = load_pickle_data(rgb_data_path)
            flow_data_info = load_pickle_data(flow_data_path)
            self.rgb_data = concat_action_segs(rgb_data_info["features"]["RGB"])
            self.flow_data = concat_action_segs(flow_data_info["features"]["Flow"])
            ids = np.array(rgb_data_info['narration_ids'])
        label_info = load_pickle_data(labels_path)
        ids_col = np.array([])
        for single_id in ids:
            id_num = int(single_id.split("_")[-1])
            verb_class = label_info.set_index('uid').loc[id_num, 'verb_class']
            ids = np.repeat(verb_class, 1)
            ids_col = np.concatenate((ids_col, ids), axis=0)
        self.labels = ids_col

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.rgb_data[index], self.flow_data[index], self.labels[index]


class RgbFlowAvgDataset(Dataset):
    def __init__(self, rgb_data_path, flow_data_path, labels_path, is_hickle):
        if is_hickle:
            rgb_data_info = load_hickle_data(rgb_data_path)
            flow_data_info = load_hickle_data(flow_data_path)
            ids = [x.decode("utf-8") for x in list(rgb_data_info["'narration_ids'"])]
            self.rgb_data = avg_action_segs(rgb_data_info["'features'"]["'RGB'"][0])
            self.flow_data = avg_action_segs(flow_data_info["'features'"]["'Flow'"][0])
        else:
            rgb_data_info = load_pickle_data(rgb_data_path)
            flow_data_info = load_pickle_data(flow_data_path)
            self.rgb_data = avg_action_segs(rgb_data_info["features"]["RGB"])
            self.flow_data = avg_action_segs(flow_data_info["features"]["Flow"])
            ids = np.array(rgb_data_info['narration_ids'])
        label_info = load_pickle_data(labels_path)
        ids_col = np.array([])
        for single_id in ids:
            id_num = int(single_id.split("_")[-1])
            verb_class = label_info.set_index('uid').loc[id_num, 'verb_class']
            ids = np.repeat(verb_class, 1)
            ids_col = np.concatenate((ids_col, ids), axis=0)
        self.labels = ids_col

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.rgb_data[index], self.flow_data[index], self.labels[index]


ft_format_class = {
    "separate": RgbFlowDataset,
    "concat": RgbFlowConcatDataset,
    "avg": RgbFlowAvgDataset
}


def load_pickle_data(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
        return data


def load_hickle_data(file_name):
    with open(file_name, 'rb') as f:
        data = hkl.load(f)
        return data


def avg_action_segs(data):
    return np.mean(data, axis=1)


def concat_action_segs(data):
    shape = data.shape
    return data.reshape(shape[0], shape[1]*1024)


def reshape_action_segs(data):
    shape = data.shape
    return data.reshape(shape[0]*shape[1], 1024)


def load_datasets(source_data_path, source_labels_path, target_data_path, target_labels_path, batch_size, source_hickle, target_hickle, action_seg_format):
    source_dataset = ft_format_class[action_seg_format](
        rgb_data_path=f"./pre_extracted_features/RGB/RGB/ek_i3d/{source_data_path}",
        flow_data_path=f"./pre_extracted_features/Flow/Flow/ek_i3d/{source_data_path}",
        labels_path=f"./label_lookup/{source_labels_path}",
        is_hickle=source_hickle
    )
    source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True)
    target_dataset = ft_format_class[action_seg_format](
        rgb_data_path=f"./pre_extracted_features/RGB/RGB/ek_i3d/{target_data_path}",
        flow_data_path=f"./pre_extracted_features/Flow/Flow/ek_i3d/{target_data_path}",
        labels_path=f"./label_lookup/{target_labels_path}",
        is_hickle=target_hickle
    )
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True)
    return source_loader, target_loader
