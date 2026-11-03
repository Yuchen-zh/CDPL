import os

import hdf5storage as hdf
import numpy as np
import scipy.io
import torch
from scipy import sparse as ss
from sklearn.preprocessing import StandardScaler, normalize
from torch.utils.data import Dataset


def get_data(hparams, dataset_path):
    try:
        data = scipy.io.loadmat(dataset_path)
    except Exception:
        data = hdf.loadmat(dataset_path)  


    mv_data = data['X'][0]
    if 'label' in data:
        labels = data['label']
    elif 'gnd' in data:
        labels = data['gnd']
    else:
        raise KeyError("Label field is neither 'label' nor 'gnd'")

    labels = labels.astype(np.int64).flatten()
    labels = labels - labels.min()
    labels = torch.from_numpy(labels).long()

    if labels.shape[0] in mv_data[0].shape:
        total_sample_num = labels.shape[0]
    elif labels.shape[1] in mv_data[0].shape:
        total_sample_num = labels.shape[0]
    else:
        raise ValueError("Cannot align the number of samples, please check the data shape")

    if total_sample_num != mv_data[0].shape[0]:
        mv_data = [v_data.T for v_data in mv_data]

    if total_sample_num != labels.shape[0]:
        labels = labels.T

    assert mv_data[0].shape[0] == labels.shape[0] == total_sample_num

    mv_data_dense = []
    for feature in mv_data:

        if ss.isspmatrix(feature):
            feature = feature.todense()

        if hparams.dataset == 'YouTubeFace20_4Views' or hparams.dataset == 'IGF-1':
            mv_data_dense.append(normalize(np.array(feature).astype(np.float32)))
        elif hparams.dataset == 'Wikipedia':
            mv_data_dense.append(np.array(feature).astype(np.float32))
        else:
            mv_data_dense.append(StandardScaler().fit_transform(np.array(feature).astype(np.float32)))

    mv_data = mv_data_dense

    indices = np.arange(total_sample_num)

    return mv_data, labels, total_sample_num, indices


class MultiViewDataset(Dataset):
    def __init__(self, mat_path, hparams, mode='train_labeled'):
        self.mv_data, self.labels, self.total_sample_num, self.indices = get_data(hparams, mat_path)
        self.d_list = [da.shape[1] for da in self.mv_data]
        self.classes_num = torch.max(self.labels).item() + 1 if len(self.labels.shape) == 1 else self.labels.shape[1]
        self.mode = mode

    def __len__(self):
        return self.total_sample_num

    def __getitem__(self, index):
        data = [torch.tensor(view[index], dtype=torch.float32) for view in self.mv_data]
        label = torch.tensor(self.labels[index], dtype=torch.long)
        idx = self.indices[index]
        if self.mode == 'train_labeled':
            return {'idx_lb': idx, 'x_lb': data, 'y_lb': label}
        elif self.mode == 'train_unlabeled':
            return {'idx_ulb': idx, 'x_ulb': data, 'y_ulb': label}
        else:
            return {'idx_lb': idx, 'x_lb': data, 'y_lb': label}


def ComDataSet(hparams, mode):
    ulb_ratio_str = f"{hparams.ulb_ratio:.2f}"
    imb_ratio_str = f"{hparams.imb_ratio:.1f}"
    dataset_base_path = os.path.join(
        hparams.data_base,
        hparams.dataset,
        'imbalance',
        ulb_ratio_str,
        hparams.ulb_imb_type,
        'gamma_' + imb_ratio_str
    )

    if hparams.dataset == 'CIFAR10_llc_with_img_fea' or hparams.dataset == 'CIFAR100':
        dataset_base_path1 = os.path.join(
            hparams.data_base,
            hparams.dataset,
            'imbalance',
            ulb_ratio_str,
        )
    else:
        dataset_base_path1 = dataset_base_path

    if mode == 'train_labeled':
        path = os.path.join(dataset_base_path, 'train_label.mat')
    elif mode == 'train_unlabeled':
        path = os.path.join(dataset_base_path, 'train_unlabel.mat')
    elif mode == 'val':
        path = os.path.join(dataset_base_path1, 'val.mat')
    elif mode == 'test':
        path = os.path.join(dataset_base_path1, 'test.mat')
    else:
        raise ValueError(f"Unknown mode: {mode}")

    dataset = MultiViewDataset(path, hparams, mode=mode)
    return dataset

