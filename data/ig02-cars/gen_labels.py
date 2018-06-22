import glob
import numpy as np
import matplotlib.pylab as plt
import matplotlib.cm as cm
import os

data_path = "cars/"
os.chdir(data_path)
for filename in glob.glob('*.image.png'):
        imgs = list()
        fn = filename.split('.')[0]
        first_it = True
        masks = glob.glob(fn+".mask.*.png" )
        if not masks:
                plt.imsave(
                        fn+".mask.png",
                        np.zeros(plt.imread(filename)[:,:,0].shape),
                        cmap=cm.gray,
                        vmin=0, vmax=1
                )
        for mask in masks:
                if first_it:
                        acc = plt.imread(mask)
                        first_it = False
                acc += plt.imread(mask)
                plt.imsave(fn+".mask.png", acc[:,:,0], cmap=cm.gray, vmin=0, vmax=1)
