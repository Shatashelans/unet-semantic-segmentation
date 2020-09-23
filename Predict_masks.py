import os
import random
import argparse

import numpy as np

import matplotlib.pyplot as plt

from tqdm import tqdm
from skimage.io import imread, imshow, imsave
from skimage.transform import resize

from keras.models import Model, load_model
from keras import backend as K

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

seed = 42
random.seed = seed
np.random.seed = seed


def dice_score(y_true, y_pred) -> float:
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    return 2 * (K.sum(y_true * y_pred)+ K.epsilon()) / (K.sum(y_true) + K.sum(y_pred) + K.epsilon())


def get_ids(path: str) -> list:
    return list(map(lambda x: x[1], os.walk(path)))[0]


def get_and_resize_images(ids: list, path: str) -> (np.array, list):
    X_test = np.zeros((len(ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    sizes_test = []
    for n, id in tqdm(enumerate(ids), total=len(ids), ncols=100):
        img = imread(os.path.join(path, id) + '/images/' + id + '.png')[:,:,:IMG_CHANNELS]
        sizes_test.append([img.shape[0], img.shape[1]])
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_test[n] = img
    return X_test, sizes_test


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--path_to_test', default='data/stage1_test', type=str,
                        help='The path to test dataset')
    parser.add_argument('--path_to_model', default='model.h5', type=str,
                        help='Path to the model')
    parser.add_argument('--path_to_output', default='test_with_predictions', type=str,
                        help='Path to the test images with predicted masks')
    args = parser.parse_args()

    test_path = args.path_to_test
    model_path = args.path_to_model

    print('\n\nModel loading ...')
    model = load_model(model_path, custom_objects={'dice_score': dice_score})
    test_ids = get_ids(test_path)
    print('\n\nGetting and resizing test images ... ')
    X_test, sizes_test = get_and_resize_images(test_ids, test_path)
    preds_test = model.predict(X_test, verbose=1)
    preds_test[preds_test >= 0.5] = 1
    preds_test[preds_test < 0.5] = 0

    test_with_predictions = args.path_to_output
    os.mkdir(test_with_predictions)
    for i in range(len(X_test)):
        folder_to_save = os.path.join(test_with_predictions, str(i))
        os.mkdir(folder_to_save)
        imsave(os.path.join(folder_to_save, 'original_image.png'), X_test[i])
        imsave(os.path.join(folder_to_save, 'mask.png'), preds_test[i])

    ix = random.randint(0, len(X_test))
    imshow(X_test[ix])
    plt.show()
    imshow(np.squeeze(preds_test[ix]))
    plt.show()


if __name__ == '__main__':
    main()
