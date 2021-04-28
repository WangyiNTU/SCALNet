import numpy as np
from scipy.ndimage import imread
import os
import random
import sys
import itertools
from density_gen import Gauss2D, Gauss2D_HM, read_image_label_fix, read_image_label_apdaptive, read_image_label_hm, \
read_image_label_3d, read_image, save_density_map, get_annoted_kneighbors
from PIL import Image
from numpy import savez_compressed as save_npy
from numpy import load as load_npy

import copy
import re

from collections import namedtuple
from src.timer import Timer

basic_config = {
    'fixed': {
        "sigma": 4.0, "f_sz": 15.0, "channels": 3, "downsize": 32
    },
    'adaptive': {
        "K": 4, "channels": 3, "downsize": 32
    },
    'hm': {
        "K": 4, "channels": 3, "downsize": 32
    },
    '3d': {
        "K":4, "S": [9,25,49,81], "channels": 3, "downsize": 32
    },
    'unlabel': {
        "channels": 3, "downsize": 32
    }
}

mode_func = {
    'fixed': read_image_label_fix,
    'adaptive': read_image_label_apdaptive,
    'hm': read_image_label_hm,
    '3d': read_image_label_3d,
    'unlabel': read_image
}

Blob = namedtuple('Blob', ('img', 'den', 'gt_count'))

class ImageDataLoader():
    def __init__(self, image_path, label_path, mode, is_preload=True, split=None, annReadFunc=None, **kwargs):

        self.image_path = image_path
        self.label_path = label_path

        self.image_files = [filename for filename in os.listdir(image_path) \
                           if os.path.isfile(os.path.join(image_path,filename))]
        self.label_files = [filename for filename in os.listdir(label_path) \
                           if os.path.isfile(os.path.join(label_path,filename))]


        self.image_files.sort(cmp=lambda x, y: cmp('_'.join(re.findall(r'\d+',x)),'_'.join(re.findall(r'\d+',y))))
        self.label_files.sort(cmp=lambda x, y: cmp('_'.join(re.findall(r'\d+',x)),'_'.join(re.findall(r'\d+',y))))

        for img, lab in zip(self.image_files, self.label_files):
            assert '_'.join(re.findall(r'\d+', img)) == '_'.join(re.findall(r'\d+',lab))

        if split != None:
            self.image_files = split(self.image_files)
            self.label_files = split(self.label_files)
        self.num_samples = len(self.image_files)
        self.mode = mode
        self.annReadFunc = annReadFunc

        self.blob_list = []

        self.fspecial = Gauss2D()
        if self.mode == 'hm':
            self.fspecial = Gauss2D_HM()
            self.fspecial2 = Gauss2D()
        self.is_preload = is_preload
        self.read_func_kwargs = kwargs

        if 'test' in kwargs.keys():
            self.test = kwargs['test']
        else:
            self.test = False

        if self.mode == 'adaptive' or self.mode == 'hm':
            self.precompute_scale()
            print("K neighbors for adaptive density map Done.")
        if self.is_preload:
            self.preload_data()
        

    def preload_data(self):
        print('Pre-loading the data. This may take a while...')

        t = Timer()
        t.tic()
        self.is_preload = False
        npy_path = os.path.join(self.image_path, self.image_files[self.num_samples-1]).replace('images','gt_npy').replace('.jpg','.npz')
        if os.path.isfile(npy_path):
            pass
        else:
            os.makedirs(self.image_path.replace('images','images_resized'))
            os.makedirs(self.image_path.replace('images','gt_npy'))
            self.blob_list = [_ for _ in range(self.num_samples)]
            for i in range(self.num_samples):
                img, den, count = self.load_index(i)
                den = den.astype(np.float32)
                image_path = os.path.join(self.image_path, self.image_files[i])
                img.save(image_path.replace('images','images_resized'),quality=100)
                save_npy(image_path.replace('images','gt_npy').replace('.jpg','.npz'),den)
                if i % 50 == 0:
                    print("loaded {}/{} samples".format(i, self.num_samples))
        duration = t.toc(average=False)
        print('Completed loading ' ,len(self.blob_list), ' files, time: ', duration)
        self.is_preload = True

    def precompute_scale(self):
        self.kneighbors = []
        for i in range(self.num_samples):
            neighbors = get_annoted_kneighbors(self.label_files[i], self.label_path, \
                            K=self.read_func_kwargs['K'], annReadFunc=self.annReadFunc)
            self.kneighbors += [neighbors]

    def load_index(self, i):
        image_file, label_file = self.image_files[i], self.label_files[i]
        if self.mode != 'adaptive' and self.mode != 'hm':
            img, den, gt_count = mode_func[self.mode](image_file, label_file, self.image_path, self.label_path, \
                                        self.fspecial.get, annReadFunc=self.annReadFunc, **self.read_func_kwargs)
        else:
            img, den, gt_count = mode_func[self.mode](image_file, label_file, self.image_path, self.label_path, \
                                        self.fspecial.get, get_gauss2 = self.fspecial2.get, annReadFunc=self.annReadFunc, kneighbors=self.kneighbors[i], \
                                        **self.read_func_kwargs)

        return Blob(img, den, gt_count)

    def query_fname(self, i):
        return self.image_files[i]

    def __getitem__(self, i):
        return self.__index__(i)

    def __index__(self, i):
        if self.is_preload:
            image_path = os.path.join(self.image_path, self.image_files[i])
            img = Image.open(image_path.replace('images','images_resized')).convert('RGB')
            den = load_npy(image_path.replace('images','gt_npy').replace('.jpg','.npz'))
            den = den['arr_0']
            count = np.sum(den[0]==1)
            blob = Blob(img, den, count)
            return blob
        else:
            return self.load_index(i)
    def __iter__(self):
        for i in range(self.num_samples):
            yield self.__index__(i)

    def get_num_samples(self):
        return self.num_samples

    def __len__(self):
        return self.num_samples


class ImageDataLoader_unlabel():
    def __init__(self, image_path, mode, split=None, **kwargs):

        self.image_path = image_path

        test_txt_path = image_path.replace('testing_data/images', 'test.txt')
        with open(test_txt_path) as f:
            file_name = []
            for line in f.readlines():
                line = line.strip().split(' ')
                file_name.append(line[0]+'.jpg')

        self.image_files = [filename for filename in os.listdir(image_path) \
                            if os.path.isfile(os.path.join(image_path, filename))]

        assert set(self.image_files) == set(file_name)
        assert self.image_files.sort() == file_name.sort()

        self.image_files.sort(cmp=lambda x, y: cmp('_'.join(re.findall(r'\d+', x)), '_'.join(re.findall(r'\d+', y))))

        if split != None:
            self.image_files = split(self.image_files)

        self.num_samples = len(self.image_files)
        self.mode = mode

        self.blob_list = []

        self.read_func_kwargs = kwargs

        if 'test' in kwargs.keys():
            self.test = kwargs['test']
        else:
            self.test = False

    def load_index(self, i):
        image_file = self.image_files[i]
        assert self.mode == 'unlabel'
        img, den, gt_count = mode_func[self.mode](image_file, self.image_path, **self.read_func_kwargs)

        return Blob(img, den, gt_count)

    def query_fname(self, i):
        return self.image_files[i]

    def __getitem__(self, i):
        return self.__index__(i)

    def __index__(self, i):
        return self.load_index(i)

    def __iter__(self):
        for i in range(self.num_samples):
            yield self.__index__(i)

    def get_num_samples(self):
        return self.num_samples

    def __len__(self):
        return self.num_samples

