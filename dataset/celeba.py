import numpy as np
import os
import pickle
import torch

from PIL import Image
from torch.utils import data

class CelebaSetsDataset(data.Dataset):
    """
    Base class for all the datasets.
    Attributes:
        data_dir: data path.
        sample_size: number of samples in conditioning dataset.
        num_classes_task: for generative models we use 1 concept per set.
        split: train/val/test
        augment: augment the sets with flipping and/or rotations.
    """ 
    def __init__(self,
                 dataset="celeba",
                 data_dir="/scratch/gigi/data/celeba/processed_celeba/",
                 sample_size=5,
                 num_classes_task=1,
                 split="train",
                 augment=False):
        super(CelebaSetsDataset, self).__init__()

        with open("/home/gigi/ns_data/celeba/map_celeba.pkl", "rb") as f:
            self.map_classes = pickle.load(f)
        self.data_dir = data_dir
        self.data_dir = "/scratch/gigi/data/celeba/processed_celeba/" #"/scratch/gigi/data/celeba/"
        self.sample_size = sample_size
        self.split = split
        self.mix = True
        self.size=64
        self.nc=3

        self.dts = {"celeba": {"size": 64, "img_cls": [20, 30], "nc": 3, "tr": 4444, "vl": 635, "ts": 1270}}
        
        classes = sorted(self.map_classes.keys())
        
        s0 = int(0.7* len(classes))
        s1 = int(0.8* len(classes))
        if self.split == "train":
            self.classes = np.array(classes[:s0])
        elif self.split == "val":
            self.classes = np.array(classes[s0:s1])
        elif self.split == "test":
            self.classes = np.array(classes[s1:])

        self.n = (len(self.classes) * 20) // self.sample_size
        self.init_sets()
            
    def init_sets(self):
        perm = np.random.permutation(len(self.classes))
        self.classes = self.classes[perm] 

    def __getitem__(self, item, lbl=None):
        samples = self.make_set(item)
        return samples

    def __len__(self):
        return self.n

    def make_set(self, item):
        item = (item % len(self.classes)) - 1
        e = self.classes[item]
        a = self.map_classes[e]
        samples = np.random.choice(a, size=self.sample_size, replace=False)
        
        lst = []
        for sample in samples:
            with Image.open(self.data_dir + sample, "r") as img:
                # [0, 1]
                img = np.array(img, dtype=np.float32) / 255.
                # [-1, 1]
                img = 2 * img - 1
                img = img.transpose(2, 0, 1)
                img = np.expand_dims(img, 0)
                lst.append(img)
        lst = np.concatenate(lst, 0)
        return lst

if __name__ == "__main__":

    dataset = CelebaSetsDataset(dataset="celeba")
    print(len(dataset))
    batch = dataset.__getitem__(0)
    print(batch.shape)
    print(batch.min(), batch.max())
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(6, 3))

    for i in range(5):
        tmp=(batch[i] + 1) / 2
        axes[i].imshow(tmp.transpose(1, 2, 0))
    fig.savefig("./_img/tmp.png")
