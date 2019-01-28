import argparse
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--mats_dir", type=str, required=True)
    parser.add_argument("--transform", action="store_true")

    parser.add_argument("--checkpoint", type=str, help="path to saved model")

    return parser.parse_args()



def gaussian_distribution(center_x, center_y, size=400):

    img = np.zeros((size, size))
    for i in  range(size):
        for j in  range(size):
            exp_fact = ((i - center_x) ** 2) / 2 + ((j - center_y) ** 2) / 2
            img[i,j] = 1 / np.sqrt(2 * np.pi) * np.exp(-exp_fact)

    img /= np.sum(img)
    return img