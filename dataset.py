from torch.utils.data import Dataset
import pickle
import numpy as np
import pandas as pd


class CustomDataset(Dataset):
    def __init__(self, data_path, labels_path, modality):
        data_info = load_data(data_path)
        label_info = load_data(labels_path)
        self.data = reshape_data(data_info, modality)
        ids = np.array(data_info['narration_ids'])
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
        return self.data[index], self.labels[index]


def load_data(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
        return data


def reshape_data(data, modality):
    shape = data['features'][modality].shape
    return data['features'][modality].reshape(shape[0]*shape[1], 1024)
