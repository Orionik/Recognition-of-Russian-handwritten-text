import matplotlib.image as img
from matplotlib import pyplot as plt
import numpy as np
import os
from PIL import Image


def get_rast(grarray):
    rast = []
    for i in grarray:
        layer = []
        for j in i:
            layer.append(np.round(j[-1]/255))
        rast.append(layer)
    return rast

def main():
    x = []
    y = []
    count = 0
    datadir = "Cyrillic\\Cyrillic\\"

    for folder in os.listdir(datadir):
        path = os.path.join(datadir, folder)
        print(folder)
        i=0
        len_dir = len(os.listdir(path))
        while i < len_dir:
            images = os.listdir(path)[i]
            pic = Image.open(os.path.join(path, images))
            new_image = pic.resize((28, 28))
            img = np.asarray(new_image)
            print(images, len_dir, i)
            x.append(get_rast(img))
            y.append(folder)
            i+=1
        
    np.save('alp_x', x)  
    np.save('alp_y', y)
    rast_x = np.load("alp_x.npy")
    rast_y = np.load("alp_y.npy")
    print(rast_x[1], rast_y[1])
    plt.imshow(rast_x[1])
    plt.show()


if __name__ == '__main__':
    main()


