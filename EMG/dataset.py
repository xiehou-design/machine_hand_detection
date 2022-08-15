import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset


class ImageDataset(Dataset):
    def __init__(self, image, label):
        super(ImageDataset, self).__init__()
        self.images = image
        self.labels = label

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, item):
        image = self.images[item]
        image = np.transpose(image, [-1, 0, 1])
        label = self.labels[item]
        return image, label
