from PIL import Image
import numpy as np 
import random as rd
from sys import stderr

def resize_to_fit(x, y, target_size):
    assert x.size == y.size
    width, height = target_size
    twidth, theight = x.size

    ratio = max(width / twidth, height / theight)
    size_after = (int(twidth * ratio) + 1, int(theight * ratio) + 1)
    x.resize(size_after, resample=Image.NEAREST)
    y.resize(size_after, resample=Image.NEAREST)


def random_crop_with_target(x, y, target_size):
    assert x.size == y.size
    width, height = target_size
    twidth, theight = x.size

    if twidth < width or theight < height:
        resize_to_fit(x, y, target_size)
        print("Warning: had to resize an image before crop", file=stderr)
        return random_crop_with_target(x, y, target_size)

    left_corner = rd.randint(0, twidth - width)
    top_corner = rd.randint(0, theight - height)
    x.crop((left_corner, top_corner, left_corner + width, top_corner + height))
    y.crop((left_corner, top_corner, left_corner + width, top_corner + height))

    return (x, y)

def random_crop(x, y, crop_ratio):
    assert x.size == y.size
    width, height = x.size
    size_before_crop = (int(width / crop_ratio), int(height / crop_ratio))
    x = x.resize(size_before_crop, resample=Image.NEAREST)
    y = y.resize(size_before_crop, resample=Image.NEAREST)

    return random_crop_with_target(x, y, (width, height))

def center_crop_with_target(x, y, target_size):
    assert x.size == y.size
    width, height = target_size
    twidth, theight = x.size

    if twidth < width or theight < height:
        resize_to_fit(x, y, target_size)
        print("Warning: had to resize an image before crop", file=stderr)
        return center_crop_with_target(x, y, target_size)

    left_corner = int(round(twidth / 2)) - int(round(width / 2))
    top_corner = int(round(theight / 2)) - int(round(height / 2))
    x.crop((left_corner, top_corner, left_corner + width, top_corner + height))
    y.crop((left_corner, top_corner, left_corner + width, top_corner + height))

    return (x, y)

def center_crop(x, y, crop_ratio):
    assert x.size == y.size
    width, height = x.size
    size_before_crop = (int(width / crop_ratio), int(height / crop_ratio))
    x = x.resize(size_before_crop, resample=Image.NEAREST)
    y = y.resize(size_before_crop, resample=Image.NEAREST)

    return center_crop_with_target(x, y, (width, height))

def class_crop_with_target(x, y, target_size, clsid):
    assert x.size == y.size
    width, height = target_size
    twidth, theight = x.size

    if twidth < width or theight < height:
        resize_to_fit(x, y, target_size)
        print("Warning: had to resize an image before crop", file=stderr)
        return class_crop_with_target(x, y, target_size, clsid)

    arr = np.array(y.getdata(), dtype=np.uint16)
    indices = list(np.where(arr == clsid))
    if len(indices) == 0:
        raise ValueError("Class not present in the given image")

    index = rd.choice(indices)
    h, w = divmod(index, twidth)
    
    left_corner = min(max(0, w - width // 2), twidth - width)
    top_corner = min(max(0, h - height // 2), theight - height)
    x.crop((left_corner, top_corner, left_corner + width, top_corner + height))
    y.crop((left_corner, top_corner, left_corner + width, top_corner + height))

    return (x, y)


def class_crop(x, y, crop_ratio, clsid):
    assert x.size == y.size
    width, height = x.size

    size_before_crop = (int(width / crop_ratio), int(height / crop_ratio))
    x = x.resize(size_before_crop, resample=Image.NEAREST)
    y = y.resize(size_before_crop, resample=Image.NEAREST)
    twidth, theight = x.size

    return class_crop_with_targe(x, y, (width, height), clsid)
    

def crop_to_square(x, y):
    assert x.size == y.size
    width, height = x.size
    s = min(width, height)

    x.crop((0, 0, s, s))
    y.crop((0, 0, s, s))

    return (x, y)


def crop_to_square_class(x, y, clsid):
    raise NotImplementedError("")


def random_horizontal_flip(x, y):
    if rd.randint(0, 1) == 1:
        x = x.transpose(Image.FLIP_LEFT_RIGHT)
        y = y.transpose(Image.FLIP_LEFT_RIGHT)
    return (x, y)


def random_scale(x, y, scales, factor=1):
    '''
    scales can be:
        - a tuple of floats (lo, hi)
        - a tuple of ints
        - a list of possible sizes
    '''
    assert x.size == y.size
    width, height = x.size

    if isinstance(scales, tuple):
        if len(tuple) != 2:
            raise ValueError("Invalid tuple passed as scales")
        lo, hi = scales
        if isinstance(lo, float) and isinstance(hi, float):
            if lo > hi:
                raise ValueError("Invalid scale scope")
            ratio = rd.uniform(lo, hi)
            twidth, theight = int(width * ratio), int(height * ratio)
        elif isinstance(lo, int) and isinstance(hi, int):
            if lo > hi:
                raise ValueError("Invalid scale scope")

            swap = False
            if height < width:
                swap = True
                width, height = height, width
            
            twidth = rd.randint(lo, hi)
            theight = int(height * (twidth / width))
            if swap:
                width, height = height, width
                twidth, theight = theight, twidth
        else:
            raise ValueError("Invalid scales argument")
    elif isinstance(scales, list):
        tsize = rd.choice(scales)
        if not isinstance(tsize, tuple) or len(tsize) != 2:
            raise ValueError("Invalid list member, expected a pair of ints")
        twidth, theight = tsize
        if not isinstance(twidth, int) or not isinstance(theight, int):
            raise ValueError("Invalid list member, expected a pair of ints")
    else:
        raise ValueError("Invalid scales argument")
        
    if factor != 1:
        raise NotImplementedError("Aligning the size of the image is not supported yet")

    tsize = (twidth, theight)
    x = x.resize(tsize, resample=Image.NEAREST)
    y = y.resize(tsize, resample=Image.NEAREST)

    return (x, y)
