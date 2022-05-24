import gzip
import os
import pickle

import numpy as np
import torch
from PIL import Image
from skimage.transform import rotate
from torch.utils import data


# adapted from durkan
class OmniglotSetsDatasetNS(data.Dataset):
    def __init__(self, 
                 dataset,
                 data_dir, 
                 sample_size, 
                 num_classes_task=1, 
                 split='train', 
                 augment=False, 
                 binarize=False):
        """
        Omniglot dataset.
        """
        self.sample_size = sample_size
        self.split = split
        self.binarize=binarize
        self.augment = augment

        self.dts = {"omniglot_ns": {"size": 28, "img_cls": 20, "nc": 1, "tr": 1000, "vl": 200, "ts": 423}}

        path = os.path.join(data_dir, dataset, 'omni_train_val_test.pkl')
        data=self.get_data(path)
        self.images, self.labels = data[split]
        
        print(self.split)
        print(self.images.shape, self.labels.shape)
        self.init_sets()
        
    @staticmethod
    def get_data(path):
        with open(path, 'rb') as file:
            data = pickle.load(file)
        return {"train": data[:2], 
                "val": data[2:4],
                "test": data[4:],
                }

    def init_sets(self):
        sets, set_labels = self.make_sets(self.images, self.labels)
        
        if self.split in ['train', 'val']:
            if self.augment:
                sets = self.augment_sets(sets)
        if self.binarize:
            sets = np.random.binomial(1, p=sets, size=sets.shape).astype(np.float32)
            
        # (batch_size, sample_size, xdim)
        sets = sets.reshape(-1, self.sample_size, 1, 28, 28)
        self.n = len(sets)
        self.data = {
        'inputs': sets,
        'targets': set_labels
        }

    def __getitem__(self, item, lbl=None):
        samples = self.data['inputs'][item]

        if self.split in ['train', 'val'] and self.augment:
            targets = np.zeros(samples.shape)
        else:
            targets = self.data['targets'][item] 
        if lbl:
            return samples, targets
        return samples

    def __len__(self):
        return self.n

    def augment_sets(self, sets):
        """Augment training sets."""
        augmented = np.copy(sets)
        augmented = augmented.reshape(-1, self.sample_size, 28, 28)
        n_sets = len(augmented)

        for s in range(n_sets):
            flip_horizontal = np.random.choice([0, 1])
            flip_vertical = np.random.choice([0, 1])
            if flip_horizontal:
                augmented[s] = augmented[s, :, :, ::-1]
            if flip_vertical:
                augmented[s] = augmented[s, :, ::-1, :]

        for s in range(n_sets):
            angle = np.random.uniform(0, 360)
            for item in range(self.sample_size):
                augmented[s, item] = rotate(augmented[s, item], angle)
        
        augmented = augmented.reshape(n_sets, self.sample_size, 28*28)
        augmented = np.concatenate([augmented, sets])
        return augmented

    @staticmethod
    def one_hot(dense_labels, num_classes):
        num_labels = len(dense_labels)
        offset = np.arange(num_labels) * num_classes
        one_hot_labels = np.zeros((num_labels, num_classes))
        one_hot_labels.flat[offset + dense_labels.ravel()] = 1
        return one_hot_labels

    def make_sets(self, images, labels):
        """
        Create sets of arbitrary size between 1 and 20. 
        The sets are composed of one class.
        """
        
        num_classes = np.max(labels) + 1
        labels = self.one_hot(labels, num_classes)
        
        n = len(images)
        perm = np.random.permutation(n)
        images = images[perm]
        labels = labels[perm]
        
        # init sets
        image_sets = []
        label_sets = []

        for j in range(num_classes):

            label = labels[:, j].astype(bool)
            num_instances_per_class = np.sum(label)
            # if num instances less than what we want (30 > 20 Omniglot max 20)
            if num_instances_per_class < self.sample_size:
                pass
            else:
                # check if sample_size is a multiple of num_instances
                remainder = num_instances_per_class % self.sample_size
                # select all images with a certain label
                image_set = images[label]
                if remainder > 0:
                    # remove samples from image_sets
                    image_set = image_set[:-remainder]
                # collect sets
                image_sets.append(image_set)
                # for Omniglot k should be 20
                k = len(image_set)
                # select only elements with certain label
                label_set = labels[label]
                # then select (k/sample_size) times the same label
                label_set = label_set[:int(k / self.sample_size)]
                label_sets.append(label_set)

        x = np.concatenate(image_sets, axis=0).reshape(-1, self.sample_size, 28*28)
        y = np.concatenate(label_sets, axis=0)
        if np.max(x) > 1:
            x /= 255

        perm = np.random.permutation(len(x))
        x = x[perm]
        y = y[perm]
        return x, y

def load_mnist(data_dir):

    def load_images(path):
        with gzip.open(path) as bytestream:
            # read meta information
            header_buffer = bytestream.read(16)
            header = np.frombuffer(header_buffer, dtype='>i4')
            magic, n, x, y = header
            # read data
            buffer = bytestream.read(x * y * n)
            data = np.frombuffer(buffer, dtype='>u1').astype(np.float32)
            data = data.reshape(n, x * y)
        return data

    def load_labels(path):
        with gzip.open(path) as bytestream:
            # read meta information
            header_buffer = bytestream.read(8)
            header = np.frombuffer(header_buffer, dtype='>i4')
            magic, n = header
            # read data
            buffer = bytestream.read(n)
            data = np.frombuffer(buffer, dtype=np.uint8).astype(np.int32)
        return data

    train_images_gz = 'train-images-idx3-ubyte.gz'
    train_labels_gz = 'train-labels-idx1-ubyte.gz'

    test_images_gz = 't10k-images-idx3-ubyte.gz'
    test_labels_gz = 't10k-labels-idx1-ubyte.gz'

    train_images = load_images(os.path.join(data_dir, train_images_gz))
    test_images = load_images(os.path.join(data_dir, test_images_gz))
    images = np.vstack((train_images, test_images)) / 255

    train_labels = load_labels(os.path.join(data_dir, train_labels_gz))
    test_labels = load_labels(os.path.join(data_dir, test_labels_gz))
    labels = np.hstack((train_labels, test_labels))

    n = len(labels)
    one_hot = np.zeros((n, 10))
    one_hot[range(n), labels] = 1
    labels = one_hot
    return images, labels


def load_mnist_test_batch(args):
    images, one_hot_labels = load_mnist(data_dir=args.mnist_data_dir)
    labels = np.argmax(one_hot_labels, axis=1)
    ixs = [np.random.choice(np.where(labels == i)[0], size=args.sample_size_test, replace=False)
           for i in range(10)]
    batch = np.array([images[ix] for ix in ixs])
    #label = np.array([labels[ix] for ix in ixs])
    return torch.from_numpy(batch).clone().repeat((args.batch_size // 10) + 1, 1, 1)[:args.batch_size]
    

if __name__ == "__main__":

    omniglot = OmniglotSetsDatasetNS("/home/gigi/ns_data/omniglot_ns/", sample_size=5)

    print(omniglot.data["inputs"].shape)
    print(omniglot.data["targets"].shape)
    print(len(omniglot))

    omniglot = OmniglotSetsDatasetNS("/home/gigi/ns_data/omniglot_ns/", sample_size=20)

    print(omniglot.data["inputs"].shape)
    print(omniglot.data["targets"].shape)
    print(len(omniglot))
