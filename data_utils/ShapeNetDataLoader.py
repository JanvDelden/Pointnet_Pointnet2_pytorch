# *_*coding:utf-8 *_*
import os
import warnings
import numpy as np
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')


class PartNormalDataset(Dataset):
    def __init__(self, root, npoints=2500, splitpath="", class_choice=None, normal_channel=False, transform=None,
                 mode="train"):
        self.transform = transform
        self.npoints = npoints
        self.root = root
        self.mode = mode
        self.splitpath = splitpath
        self.normal_channel = normal_channel
        dir_point = os.path.join(root, "01234/")
        paths = np.array(os.listdir(dir_point))
        self.datapath = []
        for path in paths:
            self.datapath.append(os.path.join(dir_point, path))
        split = np.load(splitpath).astype(np.int32)
        self.datapath =np.array(self.datapath)
        self.datapath = self.datapath[split]

    def __getitem__(self, index):
        path = self.datapath[index]
        cls = np.array([0]).astype(np.int32)
        data = np.load(path).astype(np.float32)
        untransformed_points = data[:, 0:3]
        point_set = data[:, 0:3]
        seg = data[:, -1].astype(np.int32)
        if self.transform:
            point_set, seg = self.transform(point_set, seg)
        if self.normal_channel:
            point_set = np.hstack([point_set, np.zeros((len(point_set), 3))])
        if len(point_set) < self.npoints:
            choice = np.random.choice(len(seg), self.npoints, replace=True)
        else:
            choice = np.random.choice(len(seg), self.npoints, replace=False)

        # resample
        point_set = point_set[choice, :]
        if self.mode == "eval":
            return point_set, cls, seg[choice], untransformed_points, untransformed_points[choice, :], seg
        return point_set, cls, seg[choice]

    def __len__(self):
        return len(self.datapath)
