import os
import pickle as pkl

import numpy as np
import torch
from torch.utils import data

class MNISTSetsDataset(data.Dataset):
    """
    MNIST with sets. 
    We use this dataset only for testing.
    """
    def __init__(self,
                 dataset="mnist",
                 data_dir="./",
                 sample_size=5,
                 num_classes_task=1,
                 split="train",
                 augment=False,
                 binarize=False):
        super(MNISTSetsDataset, self).__init__()

        self.data_dir = data_dir
        self.sample_size = sample_size
        self.binarize = binarize
        self.split = split
        self.n_cls = 10
        self.size=28
        self.nc=1
        
        data_dir = os.path.join(self.data_dir, "mnist_processed/MNIST/processed/")
        if self.split == "train":
            train = torch.load(data_dir + "training.pt")
            train_img, train_lbl = train
            s = int(0.9 * len(train_lbl))
            if self.split == "val":
                self.images = train_img[s:]
                self.labels = train_lbl[s:]
            else:
                self.images = train_img[:s]
                self.labels = train_lbl[:s]

        elif self.split == "test":
            test = torch.load(data_dir + "test.pt")
            test_img, test_lbl = test
            self.images = test_img
            self.labels = test_lbl

        dct={i: [] for i in range(self.n_cls)}
        for c in dct:
            dct[c] = self.images[self.labels == c] 
        self.data = dct

        self.n = len(self.labels) // self.sample_size
        
        self.init_sets() 

    def init_sets(self):
        pass

    def __getitem__(self, item, lbl=None):
        samples, targets = self.make_set(item)
        samples = samples / 255.
        if self.binarize:
            samples = samples.bernoulli()
        if lbl:
            return samples, targets
        return samples

    def __len__(self):
        return self.n

    def make_sets_clsf(self, sample_size=5):
        """
        Sets for downstream tasks.
        """
        conditioning_sets = {}
        digits = [i for i in range(self.n_cls)]

        for d in digits:
            a = self.data[d]
            ix = np.random.randint(0, len(a), sample_size)
            samples = a[ix]
            samples = samples.unsqueeze(1)
            samples = samples / 255.
            conditioning_sets[d] = samples
        return conditioning_sets

    def make_set(self, item):
        item = (item % self.n_cls)
        a = self.data[item]
        ix = np.random.randint(0, len(a), self.sample_size)
        samples = a[ix]
        samples = samples.unsqueeze(1)
        labels = torch.ones(samples.size()) * item
        return samples, labels

if __name__ == "__main__":
    # from torchvision.datasets import MNIST
    # mnist = MNIST(root="/home/gigi/ns_data/mnist_processed/", download=True)

    dataset = MNISTSetsDataset(data_dir="/home/gigi/ns_data")

    samples = dataset.__getitem__(2)
    print(samples)
    print(samples.size())
    print(samples.min(), samples.max())