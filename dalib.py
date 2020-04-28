from PIL import Image
import numpy as np

from sampling import CachedGenerator, DynamicGenerator, get_paths
import tools


def ImageAugmentator(pathlike, *, batch_size, target_size, num_classes, 
                     sampling="random", flip=False, scales=None, crop="center",
                     preprocessing_function=None, void_pixel=None, change_void_pixel=None, 
                     x_dir="img", y_dir="cls", cache=True):
    files = get_paths(pathlike, x_dir, y_dir)
    gen = CachedGenerator if cache else DynamicGenerator
    gen = gen(files, batch_size, num_classes, void_pixel)

    if sampling == "random":
        batch_getter = gen.get_random_batch
    elif sampling == "class":
        batch_getter = gen.get_class_batch
    else:
        raise ValueError("Sampling must be either 'random' or 'class'")

    if crop == "center":
        crop_fun = tools.center_crop_with_target
    elif crop == "random":
        crop_fun = tools.random_crop_with_target
    elif crop == "class":
        if sampling != "class":
            raise ValueError("Class crop mode is only available with 'class' sampling")
        crop_fun = tools.class_crop_with_target
    else:
        raise ValueError("Invalid crop mode")
    

    while True:
        batch, classes = batch_getter()

        if flip:
            batch = [tools.random_horizontal_flip(x, y) for x, y in batch]

        if scales:
            batch = [tools.random_scale(x, y, scales) for x, y in batch]

        if crop == "class":
            batch = [crop_fun(x, y, target_size, c) for (x, y), c in zip(batch, classes)]
        else:
            batch = [crop_fun(x, y, target_size) for x, y in batch]

        assert all(img[0].size == batch[0][0].size for img in batch)
        xdata = np.array([np.array(img, dtype=np.float32) for img, _ in batch])
        ydata = np.array([np.array(img, dtype=np.uint16) for _, img in batch])
        
        if preprocessing_function:
            xdata = preprocessing_function(xdata)
        
        if void_pixel and change_void_pixel:
            src, dest = void_pixel, change_void_pixel
            ydata[ydata == src] = dest
        
        yield (xdata, ydata)

if __name__ == "__main__":
    import matplotlib.pyplot as plt 

    gen = ImageAugmentator("static/", batch_size=2, target_size=(224, 224), num_classes=21, 
                           sampling="random", flip=True, scales=(224, 500), crop="random",
                           void_pixel=255)
    for i, batch in enumerate(gen):
        if i == 3: break
        x, y = batch
        assert x.shape[0] == 2
        plt.subplot(3, 4, i * 4 + 1)
        plt.imshow(x[0].astype(np.uint8))
        plt.subplot(3, 4, i * 4 + 2)
        plt.imshow(y[0])
        plt.subplot(3, 4, i * 4 + 3)
        plt.imshow(x[1].astype(np.uint8))
        plt.subplot(3, 4, i * 4 + 4)
        plt.imshow(y[1])
    plt.show()
