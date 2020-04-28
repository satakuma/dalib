import numpy as np 
import random as rd 

from PIL import Image
from pathlib import Path
from itertools import chain
from sys import stderr


def get_paths(pathlike, x_dir, y_dir):
    path = Path(pathlike)
    if not path.is_dir():
        raise ValueError("Path is not an existing directory")
    xpath = path / x_dir
    ypath = path / y_dir

    if not xpath.is_dir() or not ypath.is_dir():
        raise ValueError("Invalid subdirectories with samples")

    xfiles = list(chain(xpath.glob("*.jpg"), xpath.glob("*.png")))
    yfiles = list(chain(ypath.glob("*.png")))

    xstems = [p.stem for p in xfiles]
    ystems = [p.stem for p in yfiles]
    
    if len(set(xstems)) < len(xstems) or len(set(ystems)) < len(ystems):
        raise ValueError("Ambigious filenames in subdirectories")
    
    mut = set(xstems) & set(ystems)
    xfiles = sorted(filter(lambda p: p.stem in mut, xfiles))
    yfiles = sorted(filter(lambda p: p.stem in mut, yfiles))
    assert len(xfiles) == len(yfiles)
    samples = list(zip(xfiles, yfiles))
    return samples
        
    
class CachedGenerator:
    def __init__(self, sample_paths, batch_size, num_classes, void_pixel):
        if len(sample_paths) < batch_size:
            raise ValueError("Batch size should not exceed the number of samples")
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.samples = []
        self.void_pixel = void_pixel
        self.class_buckets = {i : [] for i in range(num_classes)}
        for x, y in sample_paths:
            ximg = Image.open(x)
            yimg = Image.open(y)
            ximg.load()
            yimg.load()
            for c in set(yimg.getdata()):
                if void_pixel and c != void_pixel:
                    self.class_buckets[c].append(len(self.samples))
            self.samples.append((ximg, yimg))
        
    def get_random_batch(self):
        return rd.sample(self.samples, self.batch_size), None

    def get_class_batch(self):
        classes = np.random.randint(low=0, high=self.num_classes, size=self.batch_size)
        batch = [rd.choice(self.class_buckets[i]) for i in classes]
        batch = [self.samples[i] for i in batch]
        return batch, classes

class DynamicGenerator:
    def __init__(self, sample_paths, batch_size, num_classes, void_pixel):
        if len(sample_paths) < batch_size:
            raise ValueError("Batch size should not exceed the number of samples")
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.sample_paths = sample_paths
        self.void_pixel = void_pixel
        self.class_buckets = {i : [] for i in range(num_classes)}
        for x, y in sample_paths:
            yimg = Image.open(y)
            for c in set(yimg.getdata()):
                if void_pixel and c != void_pixel:
                    self.class_buckets[c].append((x, y))
        
    def get_random_batch(self):
        samples = rd.sample(self.sample_paths, self.batch_size)
        batch = []
        for x, y in samples:
            ximg = Image.open(x)
            yimg = Image.open(y)
            ximg.load()
            yimg.load()
            batch.append((ximg, yimg))
        return batch, None
    
    def get_class_batch(self):
        classes = np.random.randint(low=0, high=self.num_classes, size=self.batch_size)
        samples = [rd.choice(self.class_buckets[i]) for i in classes]
        batch = []
        for x, y in samples:
            ximg = Image.open(x)
            yimg = Image.open(y)
            ximg.load()
            yimg.load()
            batch.append((ximg, yimg))
        return batch, classes
