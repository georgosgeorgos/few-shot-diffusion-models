import gzip
import numpy as np
import os
import pickle
import torch

from PIL import Image
from skimage.transform import rotate
from torch.utils import data


class ImageNetDataset(data.Dataset):
    def __init__(self,
                 data_dir,
                 sample_size,
                 num_classes_task=1,
                 split='train',
                 augment=False,):
        """
        ImageNet32x32_oord. 
        Potentially you can use the test set for few-shot transfer.
        """
        self.sample_size = sample_size
        self.split = split
        self.augment = augment
        self.size = 32
        self.nc = 3
        #path = os.path.join(data_dir, 'omniglot_back_eval/')
        self.images = self.get_data(data_dir, split)

    @staticmethod
    def get_data(data_dir, split):
        path = os.path.join(data_dir, "imagenet32-" + split + ".npy")
        file = np.load(path)
        return file

    def __getitem__(self, item):
        samples = self.images[item]
        return samples

    def __len__(self):
        return self.images.shape[0]


if __name__ == "__main__":

    dataset = ImageNetSetsDataset(
        "/home/gigi/ns_data/imagenet32_oord", sample_size=5, augment=True)
    print(dataset.data["inputs"].shape)
    print(dataset.data["targets"].shape)
    print(len(dataset))
