import os
import cv2
import torch

import nibabel as nib
import numpy as np

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from .config import TRAIN_DATASET_PATH, VOLUME_SLICES,\
                    VOLUME_START_AT, IMG_SIZE, N_CHANNELS,\
                    BATCH_SIZE



def split_data():
    directories = [f.path for f in os.scandir(TRAIN_DATASET_PATH) if f.is_dir()]
    data = []
    for i in range(len(directories)):
        data.append(directories[i][directories[i].rfind('/')+1:])

    train_test_data, val_data = train_test_split(data, test_size=0.2, shuffle=True)
    train_data, test_data = train_test_split(train_test_data, test_size=0.15, shuffle=True)

    return train_data, val_data, test_data


class MRIDataset(Dataset):
    def __init__(self, list_IDs, dim=(IMG_SIZE, IMG_SIZE), n_channels=N_CHANNELS, shuffle=True):
        self.dim = dim
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle

        self.samples = []
        for ID in list_IDs:
            for slice_idx in range(VOLUME_SLICES):
                self.samples.append((ID, slice_idx))

        if self.shuffle:
            np.random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        patient_id, slice_idx = self.samples[idx]

        X, y = self._load_sample(patient_id, slice_idx)

        return X, y

    def _load_sample(self, patient_id, slice_idx):
        patient_path = os.path.join(TRAIN_DATASET_PATH, patient_id)

        data_path = os.path.join(patient_path, f'{patient_id}_flair.nii')
        flair = nib.load(data_path).get_fdata()

        data_path = os.path.join(patient_path, f'{patient_id}_t1ce.nii')
        t1ce = nib.load(data_path).get_fdata()

        data_path = os.path.join(patient_path, f'{patient_id}_seg.nii')
        seg = nib.load(data_path).get_fdata()

        slice_pos = slice_idx + VOLUME_START_AT

        X = np.zeros((*self.dim, self.n_channels))
        X[:, :, 0] = cv2.resize(flair[:, :, slice_pos], self.dim)
        X[:, :, 1] = cv2.resize(t1ce[:, :, slice_pos], self.dim)

        y_slice = seg[:, :, slice_pos]

        y_slice[y_slice == 4] = 3

        y_one_hot = np.eye(4)[y_slice.astype(int)]

        y_resized = np.zeros((*self.dim, 4))
        for c in range(4):
            y_resized[:, :, c] = cv2.resize(y_one_hot[:, :, c], self.dim)

        X_max = np.max(X)
        if X_max > 0:
            X = X / X_max

        X = torch.FloatTensor(X).permute(2, 0, 1)
        y = torch.FloatTensor(y_resized).permute(2, 0, 1)

        return X, y



def get_dataset(data):
    dataset = MRIDataset(data)
    return dataset


def get_dataloaders(train_data, val_data, test_data):
    train_dataset = MRIDataset(train_data)
    val_dataset = MRIDataset(val_data)
    test_dataset = MRIDataset(test_data)

    pin_memory = True if torch.cuda.is_available() else False

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              pin_memory=pin_memory)
    val_dataloader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              pin_memory=pin_memory)
    test_dataloader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              pin_memory=pin_memory)
    
    return train_dataloader, val_dataloader, test_dataloader
