import skimage
import numpy as np


def path(file_path):
    im = skimage.io.imread(file_path, as_gray=True)
    return preprocess(np.asarray(im))


def array(array):
    return preprocess(array)


def preprocess(im):
    img_width = 160
    img_height = 320

    thresh = skimage.filters.threshold_otsu(im)
    binary = im > thresh
    binary = skimage.util.invert(binary)
    rescaled = skimage.transform.resize(binary, (img_width, img_height))
    skeleton = skimage.morphology.skeletonize(rescaled)
    return skeleton.astype("float32")
