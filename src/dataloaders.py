import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from PIL import Image
from torch.utils.data import Dataset


class FacesWith3DCoords(Dataset):
    """
    Images are in 300W-3D/<name>/*.jpg
    Coords are in 300W-3D-Face/<name>/*.mat
    """
    
    
    def __init__(self, images_dir: str, mats_dir: str):
        self.images, self.mats = [], []
        
        for i in os.listdir(images_dir):
            if i.endswith(".jpg"):
                self.images.append(os.path.join(images_dir, i))
                self.mats.append(os.path.join(mats_dir, i.split(".")[0] + ".mat"))
        
        assert len(self.images) == len(self.mats)
    
    
    def __getitem__(self, index):
        assert 0 <= index < len(self.images)
        
        img = Image.open(self.images[index])
        mat = scipy.io.loadmat(self.mats[index])['Fitted_Face']
        
        return img, mat
    
    
    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    import sys
    
    d = FacesWith3DCoords(images_dir=sys.argv[1], mats_dir=sys.argv[2])
    
    i, m = d[np.random.randint(len(d))]
    print(m)
    
    plt.imshow(i)
    plt.show()
