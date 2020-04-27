from PIL import Image
import numpy as np

import sampling
import tools


def ImageAugmentator(pathlike, *, batch_size, target_size, num_classes, 
                     sampling="random", flip=False, scales=None, crop="center",
                     preprocessing_function=None, change_void_pixel=None, 
                     x_dir="img", y_dir="cls", cache=True):
    files = sampling.get_paths(pathlike, x_dir, y_dir)
    gen = sampling.CachedGenerator if cache else sampling.DynamicGenerator
    gen = gen(files, batch_size, num_classes)
    if sampling == "random":
        batch_getter = gen.get_random_batch
    elif sampling == "class":
        batch_getter = gen.get_class_batch
    else:
        raise ValueError("Sampling must be either 'random' or 'class'")

    while True:
        batch, classes = batch_getter()
        if flip:
            batch = [tools.random_horizontal_flip(x, y) for x, y in batch]
        if scales:
            # assert isinstance(scales, tuple) and isinstance(scales[0], int)
            batch = [tools.random_scale(x, y, scales) for x, y in batch]
        if crop:
            ...

        assert all(img.size == batch[0].size for img in batch)
        xdata = np.array([np.array(img, dtype=np.float32) for img, _ in batch])
        ydata = np.array([np.array(img, dtype=np.uint16) for _, img in batch])
        
        if preprocessing_function:
            xdata = preprocessing_function(xdata)
        
        if change_void_pixel:
            src, dest = change_void_pixel
            ydata[ydata == src] = dest
        
        yield (xdata, ydata)
