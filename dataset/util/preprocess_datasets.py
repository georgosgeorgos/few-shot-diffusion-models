import os
import pickle
from PIL import Image
import h5py
import json

# dowload raw data
# from torchmeta.datasets import Omniglot, DoubleMNIST, TripleMNIST, CUB, FC100
# dataset = Omniglot("./", meta_split='train', download=True)


if __name__ == "__main__":
    import numpy as np
    import pickle
    import io

    def process_cifar100_dataset(split, name=""):
        """
        Format h5py.
        Each key is a class. Values are all the images with that class.
        """
        data_file = h5py.File("./" + name + "/" + "data.hdf5", 'r')
        
        data_resized = {}
        with open("./" + name + "/fc100/" + split + "_labels.json", 'r') as f:
            classes = json.load(f)
        
        for cl in classes:
            print(cl)
            data = data_file[cl[0]]
            images = data[cl[1]]
            
            tmp = []
            for i in range(images.shape[0]):
                img = Image.fromarray(images[i])
                tmp.append(np.array(img))

            tmp = np.stack(tmp, 0)
            data_resized[cl[1]] = tmp

        with open(split + "_" + name + ".pkl", 'wb') as f:
            pickle.dump(data_resized, f)

    def process_rgb_dataset(split, name="", size=64):
        """
        Minimagenet and CUB.
        Format h5py.
        Each key is a class. Values are all the samples for that class.
        """
        data_file = h5py.File("./" + name + "/" + split + "_data.hdf5", 'r')
        data = data_file['datasets']
        
        data_resized = {}
        classes = data.keys()

        for cl in classes:
            print(cl)
            images = data[cl]
            
            tmp = []
            for i in range(images.shape[0]):
                if name == "cub":
                    img = Image.open(io.BytesIO(images[i])).convert('RGB')
                else:
                    img = Image.fromarray(images[i])
                
                img_resized = img.resize((size, size), Image.BOX)
                tmp.append(np.array(img_resized))
                
            tmp = np.stack(tmp, 0)
            data_resized[cl] = tmp

        with open(split + "_" + name + ".pkl", 'wb') as f:
            pickle.dump(data_resized, f)

    
    def process_omniglot_dataset(split, name=""):
        """
        Format h5py.
        Omniglot different format than the other binary datasets.
        """
        data_file = h5py.File("./" + name + "/" + "data.hdf5", 'r')
        if split == "train":
            data = data_file["images_background"]
        elif split == "test":
            data = data_file["images_evaluation"]
        else:
            print("No validation for Omniglot")

        data_resized = {}
        print(data.keys())
        alphabets = data.keys()

        c = 0
        for alphabet in alphabets:
            data_alphabet = data[alphabet]
        
            classes = data_alphabet.keys()
            for cl in classes:
                print(cl)
                images = data_alphabet[cl]
                
                tmp = []
                for i in range(images.shape[0]):
                    img = Image.fromarray(images[i])
                    img_resized = img.resize((28, 28), Image.BOX)
                    tmp.append(np.array(img_resized))
                    
                tmp = np.stack(tmp, 0)
                data_resized[c] = tmp
                c += 1

        with open(split + "_" + name + ".pkl", 'wb') as f:
            pickle.dump(data_resized, f)

    def process_binary_dataset(split, name=""):
        """
        Format h5py.
        Process doubleMNIST and tripleMNIST.
        Each key is a class. Values are all the images with that class.
        For binary datasets, need to convert it back. 
        """
        data_file = h5py.File("./" + name + "/" + split + "_data.hdf5", 'r')
        data = data_file['datasets']
        
        data_resized = {}
        print(data.keys())
        
        classes = data.keys()
        for cl in classes:
            print(cl)
            images = data[cl]
            
            tmp = []
            for i in range(images.shape[0]):
                img = Image.open(io.BytesIO(images[i])).convert('L')#.convert('L')
                img_resized = img.resize((28, 28), Image.BOX)
                tmp.append(np.array(img_resized))
            tmp = np.stack(tmp, 0)
            data_resized[cl] = tmp

        with open(split + "_" + name + ".pkl", 'wb') as f:
            pickle.dump(data_resized, f)

    #for name in ["doubelmnist", "triplemnist"]:
    name = "cifar100"
    #size = 64
    if name == "omniglot":
        process_omniglot_dataset('train', name)
        process_omniglot_dataset('test', name)
    elif name in ["doubelmnist", "triplemnist"]:
        process_binary_dataset('train', name)
        process_binary_dataset('val', name)
        process_binary_dataset('test', name)
    elif name in ["minimagenet", "cub"]:
        process_rgb_dataset('train', name, size)
        process_rgb_dataset('val', name, size)
        process_rgb_dataset('test', name, size)
    elif name in ["cifar100"]:
        process_cifar100_dataset('train', name)
        process_cifar100_dataset('val', name)
        process_cifar100_dataset('test', name)

    # print("test")
    # process_minimagenet('test')
    # with open("train_minimagenet.pkl", 'rb') as f:
    #     data = pickle.load(f)
    
    # tmp = data['n03476684'][0]
    # print(tmp / 255.)
    