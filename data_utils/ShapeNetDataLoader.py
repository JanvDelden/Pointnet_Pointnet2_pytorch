# *_*coding:utf-8 *_*
import os
import json
import warnings
import numpy as np
from torch.utils.data import Dataset
import random
warnings.filterwarnings('ignore')


def pc_normalize(pc):
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


class PartNormalDataset(Dataset):
    def __init__(self, root, npoints=2500, splitpath, class_choice=None, normal_channel=False,
                 mode="train"):
        self.npoints = npoints
        self.root = root
        self.mode = mode
        self.splitpath = splitpath
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.normal_channel = normal_channel

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        if not class_choice is  None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}
        # print(self.cat)

        self.meta = {}

        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = np.array(sorted(os.listdir(dir_point)))

            # OUR CODE
            # outsource the choice of trees to easily choose random or deterministic behavior
            indices = np.load(self.splitpath)
            fns = fns[indices]

            # print(os.path.basename(fns))
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.npy'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]

        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'Tree': [0, 1]}

        # for cat in sorted(self.seg_classes.keys()):
        #     print(cat, self.seg_classes[cat])

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000

    def __getitem__(self, index):
        fn = self.datapath[index]
        cat = self.datapath[index][0]
        cls = self.classes[cat]
        cls = np.array([cls]).astype(np.int32)
        data = np.load(fn[1]).astype(np.float32)
        untransformed_points = data[:, 0:3]
        if not self.normal_channel:
            point_set = data[:, 0:3]
        else:
            point_set = data[:, 0:3]
            point_set = np.hstack([point_set, np.zeros((len(point_set), 3))])
        seg = data[:, -1].astype(np.int32)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        # resample
        point_set = point_set[choice, :]
        seg = seg[choice]
        if self.mode == "eval":
            return point_set, cls, seg, untransformed_points, untransformed_points[choice, :]
        return point_set, cls, seg

    def __len__(self):
        return len(self.datapath)