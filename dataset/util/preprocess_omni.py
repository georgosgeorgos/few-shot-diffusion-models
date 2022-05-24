import argparse
import numpy as np
import os
import pickle

from scipy.io import loadmat

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, default="/home/gigi/ns_data/omniglot_ns")

def _load():
    # load data
    file = os.path.join(args.data_dir, 'chardata.mat')
    data = loadmat(file)

    # data is in train/test split so read separately
    train_images = data['data'].astype(np.float32).T
    train_alphabets = np.argmax(data['target'].astype(np.float32).T, axis=1)
    train_characters = data['targetchar'].astype(np.float32)

    test_images = data['testdata'].astype(np.float32).T
    test_alphabets = np.argmax(data['testtarget'].astype(np.float32).T, axis=1)
    test_characters = data['testtargetchar'].astype(np.float32)

    # combine train and test data
    images = np.concatenate([train_images, test_images], axis=0)
    alphabets = np.concatenate([train_alphabets, test_alphabets], axis=0)
    characters = np.concatenate([np.ravel(train_characters),
                                 np.ravel(test_characters)], axis=0)
    data = (images, alphabets, characters)
    return data

# def load():
#     # load data
#     file = os.path.join(args.data_dir, 'chardata.mat')
#     data = loadmat(file)

#     # data is in train/test split so read separately
#     train_images = data['data'].astype(np.float32).T
#     train_alphabets = np.argmax(data['target'].astype(np.float32).T, axis=1)
#     train_characters = data['targetchar'].astype(np.float32)

#     tr_images = np.concatenate([train_images], axis=0)
#     tr_alphabets = np.concatenate([train_alphabets], axis=0)
#     tr_characters = np.concatenate([np.ravel(train_characters)], axis=0)

#     test_images = data['testdata'].astype(np.float32).T
#     test_alphabets = np.argmax(data['testtarget'].astype(np.float32).T, axis=1)
#     test_characters = data['testtargetchar'].astype(np.float32)

#     # combine train and test data
#     ts_images = np.concatenate([test_images], axis=0)
#     ts_alphabets = np.concatenate([test_alphabets], axis=0)
#     ts_characters = np.concatenate([np.ravel(test_characters)], axis=0)
    
#     tr_data = (tr_images, tr_alphabets, tr_characters)
#     ts_data = (ts_images, ts_alphabets, ts_characters)
#     return tr_data, ts_data

# def modify(data):
#     # We don't care about alphabets, so combine all alphabets
#     # into a single character ID.
#     # First collect all unique (alphabet, character) pairs.
#     images, alphabets, characters = data
#     unique_alphabet_character_pairs = list(set(zip(alphabets, characters)))

#     # Now assign each pair an ID
#     ids = np.asarray([unique_alphabet_character_pairs.index((alphabet, character))
#                       for (alphabet, character) in zip(alphabets, characters)])

#     # Now split into train(1200)/val(323)/test(100) by character
#     # train_images = images[ids < 1200]
#     # train_labels = ids[ids < 1200]
#     # # val_images = images[(1200 <= ids) * (ids < 1523)]
#     # val_labels = ids[(1200 <= ids) * (ids < 1523)]
#     # test_images = images[1523 <= ids]
#     # test_labels = ids[1523 <= ids]

#     # split_data = (train_images, train_labels, 
#     #               val_images, val_labels, 
#     #               test_images, test_labels)
#     return (images, ids)

# def _modify(data):
#     # We don't care about alphabets, so combine all alphabets
#     # into a single character ID.
#     # First collect all unique (alphabet, character) pairs.
#     images, alphabets, characters = data
#     unique_alphabet_character_pairs = list(set(zip(alphabets, characters)))

#     # Now assign each pair an ID
#     ids = np.asarray([unique_alphabet_character_pairs.index((alphabet, character))
#                       for (alphabet, character) in zip(alphabets, characters)])

#     print(ids.shape)
#     print(images.shape)

#     # Now split into train(1200)/val(323)/test(100) by character
#     train_images = images[ids < 1200]
#     train_labels = ids[ids < 1200]

#     test_images = images[1200 <= ids]
#     test_labels = ids[1200 <= ids]
#     #val_images = images[(1200 <= ids) * (ids < 1523)]
#     #val_labels = ids[(1200 <= ids) * (ids < 1523)]
#     #test_images = images[1523 <= ids]
#     #test_labels = ids[1523 <= ids]

#     print(train_images.shape, test_images.shape)
#     print(train_labels.shape, test_labels.shape)
#     split_data = (train_images, train_labels, 
#                   test_images, test_labels)
#     return split_data

def _modify(data):
    # We don't care about alphabets, so combine all alphabets
    # into a single character ID.
    # First collect all unique (alphabet, character) pairs.
    images, alphabets, characters = data
    unique_alphabet_character_pairs = list(set(zip(alphabets, characters)))

    # Now assign each pair an ID
    ids = np.asarray([unique_alphabet_character_pairs.index((alphabet, character))
                      for (alphabet, character) in zip(alphabets, characters)])

    print(ids.shape)
    print(images.shape)

    # Now split into train(1000)/val(200)/test(460) by character
    train_images = images[ids < 1000]
    train_labels = ids[ids < 1000]
    
    val_images = images[(1000 <= ids) * (ids < 1200)]
    val_labels = ids[(1000 <= ids) * (ids < 1200)]

    test_images = images[1200 <= ids]
    test_labels = ids[1200 <= ids]
    
    print(train_images.shape, val_images.shape, test_images.shape)
    print(train_labels.shape, val_labels.shape, test_labels.shape)
    split_data = (train_images, train_labels,
                  val_images, val_labels, 
                  test_images, test_labels)
    return split_data



# def main():
#     tr, ts = load()
#     tr_data = modify(tr)
#     ts_data = modify(ts)

#     tr_img, tr_lbl = tr_data
#     ts_img, ts_lbl = ts_data
#     print(tr_img.shape, tr_lbl.shape, ts_img.shape, ts_lbl.shape)

#     data = (tr_img, tr_lbl, ts_img, ts_lbl)
#     #save(data)

def save(data):
    savepath = os.path.join(args.data_dir, 'omni_train_val_test.pkl')
    with open(savepath, 'wb') as file:
        pickle.dump(data, file)


def _main():
    data = _load()
    modified_data = _modify(data)
    save(modified_data)

if __name__ == '__main__':
    args = parser.parse_args()
    assert (args.data_dir is not None) and (os.path.isdir(args.data_dir))
    _main()