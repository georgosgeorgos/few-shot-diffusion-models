import os
import numpy as np
import pickle
import random
import matplotlib.pyplot as plt

data_dir = "/home/gigi/ns_data"

for dataset in ["cifar100", "cifar100mix", "minimagenet"]:
    split="train"
    path = os.path.join(data_dir, dataset, split + "_" + dataset + ".pkl")
    with open(path, 'rb') as f:
        file_train = pickle.load(f)

    split="val"
    path = os.path.join(data_dir, dataset, split + "_" + dataset + ".pkl")
    with open(path, 'rb') as f:
        file_val = pickle.load(f)

    split="test"
    path = os.path.join(data_dir, dataset, split + "_" + dataset + ".pkl")
    with open(path, 'rb') as f:
        file_test = pickle.load(f)


    dct = {"train": file_train, "val": file_val, "test": file_test}

# dct={}
# for k in file_train:
#     dct[k] = 0
# for k in file_val:
#     dct[k] = 0
# for k in file_test:
#     dct[k] = 0

# print(len(dct))
# lst = list(dct.keys())

# random.shuffle(lst)
# print()
# print(lst[:60])
# print()
# print(lst[60:80])
# print()
# print(lst[80:])


# train = {}
# for k in lst[:60]:
#     if k in file_train:
#         train[k] = file_train[k]
#     elif k in file_val:
#         train[k] = file_val[k]
#     else:
#         train[k] = file_test[k]

# val = {}
# for k in lst[60:80]:
#     if k in file_train:
#         val[k] = file_train[k]
#     elif k in file_val:
#         val[k] = file_val[k]
#     else:
#         val[k] = file_test[k]
        

# test = {}
# for k in lst[80:]:
#     if k in file_train:
#         test[k] = file_train[k]
#     elif k in file_val:
#         test[k] = file_val[k]
#     else:
#         test[k] = file_test[k]

        
    l = {"train": [60, 60], "val": [20, 20], "test": [20, 20]}

    for split in ["train", "val", "test"]:

        n=l[split][0]
        if dataset == "minimagenet" and split == "val":
            n=16

        m=10+1
        fig, ax = plt.subplots(n, m, figsize=(10, l[split][1]))
        plt.tick_params(
            axis="both",
            which="both",
            bottom="off",
            top="off",
            labelbottom="off",
            right="off",
            left="off",
            labelleft="off",
        )

        for i, k in enumerate(dct[split]):
            print(i)
            if i >= n:
                break
            ax[i, -1].text(x=0, y=0, s=k)
            for j in range(m-1):
                ax[i, j].imshow(dct[split][k][j])

        for i in range(n):
            for j in range(m):
                ax[i, j].get_xaxis().set_ticks([])
                ax[i, j].get_yaxis().set_ticks([])
                ax[i, j].axis("off")

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig("./_vis/" + dataset + "_"  + split + ".png", bbox_inches='tight')
        plt.savefig("./_vis/" + dataset + "_" + split + ".pdf", format="pdf", bbox_inches='tight', dpi=100)
