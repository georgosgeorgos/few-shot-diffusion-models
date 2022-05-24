import gzip
import numpy as np
import os
import pickle
import torch

from PIL import Image
from skimage.transform import rotate
from torch.utils import data


class BaseSetsDataset(data.Dataset):
    """
    Base class for datasets stored in .pkl
    Attributes:
        data_dir: data path.
        sample_size: number of samples in conditioning dataset.
        num_classes_task: for generative models we use 1 concept per set.
        split: train/val/test
        augment: augment the sets with flipping and/or rotations (only for binary datasets).
    """ 
    def __init__(self,
                 dataset,
                 data_dir,
                 sample_size,
                 num_classes_task=1,
                 split='train',
                 augment=False,
                 norm=False,
                 binarize=False):
        super(BaseSetsDataset, self).__init__()

        self.dts = {"omniglot_back_eval": {"size": 28, "img_cls": 20, "nc": 1, "tr": 964, "vl": 97, "ts": 659},
                    "omniglot_random": {"size": 28, "img_cls": 20, "nc": 1, "tr": 964, "vl": 97, "ts": 659},
                    "doublemnist":  {"size": 28, "img_cls": 1000, "nc": 1, "tr":64, "vl": 16, "ts": 20},
                    "triplemnist":  {"size": 28, "img_cls": 1000, "nc": 1, "tr": 640, "vl": 160, "ts": 200},
                    "minimagenet":  {"size": 32, "img_cls": 600, "nc": 3, "tr": 64, "vl": 16 , "ts": 20}, 
                                     #"shift": -112.6077, "scale": 1. / 68.315056},
                    "cifar100":     {"size": 32, "img_cls": 600, "nc": 3, "tr":60, "vl":20, "ts": 20},
                    "cifar100mix":     {"size": 32, "img_cls": 600, "nc": 3, "tr":60, "vl":20, "ts": 20},
                    "cub":          {"size": 64, "img_cls": 60, "nc": 3, "tr": 100, "val": 50, "ts": 50},
                    }
        self.data_dir = data_dir
        self.split = split
        self.dataset = dataset
        self.augment = augment
        self.binarize = binarize
        self.sample_size = sample_size + 1
        self.norm=norm
        
        self.nc = self.dts[dataset]["nc"]
        self.size = self.dts[dataset]["size"]
        self.img_cls = self.dts[dataset]["img_cls"]
        self.n_bits = 8

        self.images, self.labels, self.map_cls = self.get_data()
        self.split_train_val()
        
        print(self.split)
        print(self.images.shape, self.labels.shape)

        self.init_sets()
        
    def init_sets(self):
        sets, set_labels = self.make_sets(self.images, self.labels)
        if self.split in ["train", "train_indistro"]:
            if self.augment:
                sets, set_labels = self.augment_sets(sets, set_labels)

        sets = sets.reshape(-1, self.sample_size, self.nc, self.size, self.size)
        self.n = len(sets)
        self.data = {
            'inputs': sets,
            'targets': set_labels
        }

    def get_data(self):
        img = []
        
        path = os.path.join(self.data_dir, self.dataset, self.split + "_" + self.dataset + ".pkl")
        with open(path, 'rb') as f:
            file = pickle.load(f)

        map_cls = {}
        for i, k in enumerate(file):

            map_cls[i] = k
            value = file[k]
            
            # if only one channel (1, img_dim, img_dim)
            if self.dataset in ["doublemnist", "triplemnist"]:
                value = np.expand_dims(value, axis=1)
            # if less than img_cls, fill residual
            residual = self.img_cls - value.shape[0]
            if residual > 0:
                value = np.vstack([value, value[:residual]])
            # if data is not rescaled between [0, 1] or [-1, 1]
            if np.max(value) > 1:
                # / 255
                value = value / (2**self.n_bits - 1)
            # (b, c, h, w)
            value = value.transpose(0, 3, 1, 2)
            img.append(value.reshape(self.img_cls, -1))

        # this works only if we have the same number of samples in each class
        img = np.array(img, dtype=np.float32)
        lbl = np.arange(img.shape[0]).reshape(-1, 1)
        lbl = lbl.repeat(self.img_cls, 1)
        return img, lbl, map_cls

    def __getitem__(self, item, lbl=None):
        samples = self.data['inputs'][item]
        # all datasets should be in [0, 1]
        if self.dataset in ['minimagenet', 'cub', 'cifar100', "celeba", "omniglot_back_eval",  "cifar100mix"]: #and self.norm:

            if self.dataset == "omniglot_back_eval":
                # dequantize
                samples = samples * 255.
                # noise [-.5, .5]
                samples = samples + (np.random.random(samples.shape) - 0.5)
                samples = samples / 255.
                samples = samples.astype(np.float32)
                
            # rescale to [-1, 1]
            samples = 2 * samples - 1
        if lbl:
            targets = self.data['targets'][item]
            return samples, targets
        return samples

    def __len__(self):
        return self.n

    def augment_sets(self, sets, sets_lbl):
        """
        Augment training sets.
        """
        augmented = np.copy(sets)
        augmented = augmented.reshape(-1, self.sample_size,
                                      self.nc, self.size, self.size)
        n_sets = len(augmented)
        # number classes for sets
        n_cls = len(sets_lbl)
        augmented_lbl = np.arange(n_cls, 2 * n_cls).reshape(-1, 1)
        augmented_lbl = augmented_lbl.repeat(self.sample_size, 1)

        # flip set
        for s in range(n_sets):
            flip_horizontal = np.random.choice([0, 1])
            flip_vertical = np.random.choice([0, 1])
            if flip_horizontal:
                augmented[s] = augmented[s, :, :, :, ::-1]
            if flip_vertical:
                augmented[s] = augmented[s, :, :, ::-1, :]
        
        # if self.dataset in ["doublemnist", "triplemnist"]:      
        #     #rotate images in set only if binary
        #     for s in range(n_sets):
        #         angle = np.random.uniform(-10, 10)
        #         for item in range(self.sample_size):
        #             augmented[s, item] = rotate(augmented[s, item], angle)

            # even if the starting images are binarized, the augmented one are not
            # augmented = np.random.binomial(1, p=augmented, size=augmented.shape).astype(np.float32)
            
        augmented = np.concatenate([augmented, sets])
        augmented_lbl = np.concatenate([augmented_lbl, sets_lbl])

        perm = np.random.permutation(len(augmented))
        augmented = augmented[perm]
        augmented_lbl = augmented_lbl[perm]
        return augmented, augmented_lbl

    def make_sets(self, images, labels):
        """
        Create sets of arbitrary size between 1 and 20. 
        The sets are composed of one class.
        """

        num_classes = np.max(labels) + 1
        
        perm = np.random.permutation(len(images))
        images = images[perm]
        labels = labels[perm]
        
        # init sets
        image_sets = images.reshape(-1, self.sample_size,
                                    self.nc, self.size, self.size)
        label_sets = labels.reshape(-1, self.sample_size)
        
        perm = np.random.permutation(len(image_sets))
        x = image_sets[perm]
        y = label_sets[perm]
        return x, y

    def split_train_val(self, ratio=0.9):
        if self.dataset in ["omniglot_back_eval"]:

            s = int(ratio* self.images.shape[0])
            if self.split == "train":
                self.images = self.images #[:s]
                self.labels = self.labels #[:s]
            
            elif self.split == "train_indistro":
                self.images = self.images[:s][:, :15]
                self.labels = self.labels[:s][:, :15]
            elif self.split == "test_indistro":
                self.images = self.images[:s][:, 15:]
                self.labels = self.labels[:s][:, 15:]

            elif self.split == "val":
                self.images = self.images[s:]
                self.labels = self.labels[s:]
                self.labels = np.arange(self.labels.shape[0]).reshape(-1, 1)
                self.labels = self.labels.repeat(self.img_cls, 1)


if __name__ == "__main__":

    dataset = BaseSetsDataset(dataset="cub", data_dir="/home/gigi/ns_data/", sample_size=5, split="val", augment=False)
    print(dataset.data["inputs"].shape)
    print(dataset.data["targets"].shape)
    print(len(dataset))

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(6, 3))

    for i in range(5):
        axes[i].imshow(dataset.data["inputs"][1][i].transpose(1, 2, 0))
    fig.savefig("./tmp.png")
