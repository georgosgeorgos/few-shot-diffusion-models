import os

from torch.utils import data
from torchvision.transforms import Compose, Resize, ToTensor

from dataset.base import BaseSetsDataset
from dataset.mnist_binary import MNISTSetsDataset
from dataset.celeba import CelebaSetsDataset
from dataset.omniglot_gmn import OmniglotSetsDatasetGMN, OmniglotSetsDatasetGMNRandom
from dataset.omniglot_ns import OmniglotSetsDatasetNS
from dataset.util.transforms import DynamicBinarize, StaticBinarize

def select_dataset(args, split):
    if split == "vis":
        split = "test"
    kwargs = {
        "dataset": args.dataset,
        "data_dir": args.data_dir,
        "sample_size": args.sample_size,
        "num_classes_task": args.num_classes,
        "split": split,
        "augment": args.augment,
    }

    if args.dataset in ["cifar100", "cifar100mix", "cub", "minimagenet", "doublemnist", "triplemnist"]:
        dataset = BaseSetsDataset(**kwargs)
    
    elif args.dataset == "celeba":
        dataset = CelebaSetsDataset(**kwargs)

    elif args.dataset == "mnist":
        kwargs["binarize"] = True #args.binarize
        dataset = MNISTSetsDataset(**kwargs)
        
    elif args.dataset == "omniglot_back_eval":
        kwargs["binarize"] = True
        dataset = OmniglotSetsDatasetGMN(**kwargs)
    elif args.dataset == "omniglot_random":
        kwargs["binarize"] = True
        dataset = OmniglotSetsDatasetGMNRandom(**kwargs)

    # omniglot used in neural statistician
    elif args.dataset == "omniglot_ns":
        kwargs["binarize"] = True
        if split in ["test", "vis"]:
            kwargs["sample_size"] = args.sample_size_test
        if split == "vis":
            kwargs["binarize"] = False
            kwargs["split"] = "test"
        dataset = OmniglotSetsDatasetNS(**kwargs)

    else:
        print("No dataset available.")
    return dataset


def create_loader(args, split, shuffle, drop_last=False):
    dataset = select_dataset(args, split)
    bs = args.batch_size
    if split in ["vis", "val", "test"]:
        bs = args.batch_size_eval
    loader = data.DataLoader(
        dataset=dataset,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=1,
        drop_last=drop_last,
    )
    while True:
        yield from loader
        
if __name__ == "__main__":
    import argparse

    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name", default="FSGM", type=str, help="readable name for run",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/home/gigi/ns_data",
        help="location of formatted Omniglot data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/gigi/ns_output",
        help="output directory for checkpoints and figures",
    )
    parser.add_argument(
        "--tag", type=str, default="", help="readable tag for interesting runs",
    )
    # dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default="omniglot_ns",
        help="select dataset (omniglot_ns, omniglot, mini_imagenet, fc100)",
    )
    # parse args
    args = parser.parse_args()

    args.dataset = "minimagenet"
    args.batch_size = 10
    args.num_classes = 1
    args.num_workers = 1
    args.sample_size = 5
    args.sample_size_test = 5

    args.binarize = False
    args.augment = True
    args.download = True
    shuffle = True
    args.split = "train"
    args.drop_last = False

    for d in [
        #"minimagenet",
        "cifar100",
        #"cub",
        #"omniglot_back_eval",
        #"omniglot_ns_trts",
        #"doublemnist",
        #"triplemnist",
    ]:
        print(d)
        args.dataset = d
        _, train_loader = create_loader(args, split="train", shuffle=True)
        print()
        _, test_loader = create_loader(args, split="test", shuffle=False)
        print(len(train_loader), len(test_loader))
        print(len(train_loader.dataset), len(test_loader.dataset))
        batch = next(iter(train_loader))
        print(batch.size())
        print(batch.min(), batch.max())
        print()
        
        import matplotlib.pyplot as plt
        tmp = batch[0][0].permute(1, 2, 0).cpu().numpy()
        tmp += 1
        tmp *= 127.5 
        tmp = tmp.astype(int)
        plt.imshow(tmp)
        plt.savefig("_img/tmp.png")
        
