"""
Code borrowed from https://github.com/POSTECH-CVLab/point-transformer/blob/master/util/transform.py
"""

import numpy as np
import torch


class Compose(object):
    """
    Composes transformations to be passed to dataset class
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, points, label):
        for t in self.transforms:
            points, label = t(points, label)
        return points, label


class ToTensor(object):
    """
    ensures correct format before returning data
    """

    def __call__(self, points, label):
        points = torch.from_numpy(points)
        if not isinstance(points, torch.FloatTensor):
            points = points.float()
            points = torch.unsqueeze(points, dim=0)
            points = torch.transpose(points, 1, 2)
        label = torch.from_numpy(label)
        if not isinstance(label, torch.FloatTensor):
            label = label.float()
        return points, label


class Normalize(object):
    """
    rescales the x, y and z values
    optionally fixed values for rescaling can be provided, otherwise the biggest distance from the zero point is used
    """

    def __init__(self, values=np.array([6.9, 6.9, 15])):
        self.values = values

    def __call__(self, points, label):
        if self.values is None:
            # this is approx. 30 for a typical chunk
            m = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
            points = points / m
        else:
            points = points / self.values
            points[:, 2] = points[:, 2] - 1
        return points, label


class RandomRotate(object):
    """
    Rotation around the z axis.
    """

    def __init__(self, angle=1):
        self.angle = angle

    def __call__(self, points, label):
        angle = np.random.uniform(-self.angle, self.angle) * np.pi
        cos, sin = np.cos(angle), np.sin(angle)
        R = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])
        points = np.dot(points, np.transpose(R))
        return points, label


class RandomScale(object):
    """
    The points are linearly scaled in size. thus a tree could become bigger or smaller.
    If Anisotropic, the x, y and z components are scaled by different values
    """

    def __init__(self, scale=[0.9, 1.1], anisotropic=False):
        self.scale = scale
        self.anisotropic = anisotropic

    def __call__(self, points, label):
        scale = np.random.uniform(self.scale[0], self.scale[1], 3 if self.anisotropic else 1)
        points *= scale
        return points, label


class RandomShift(object):
    """
    Shifts all points a given maximum distance
    """

    def __init__(self, shift=[0.2, 0.2, 0]):
        self.shift = shift

    def __call__(self, points, label):
        shift_x = np.random.uniform(-self.shift[0], self.shift[0])
        shift_y = np.random.uniform(-self.shift[1], self.shift[1])
        shift_z = np.random.uniform(-self.shift[2], self.shift[2])
        points += [shift_x, shift_y, shift_z]
        return points, label


class RandomFlip(object):
    """
    mirrors all points on the x and on the y axis
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, points, label):
        if np.random.rand() < self.p:
            points[:, 0] = -points[:, 0]
        if np.random.rand() < self.p:
            points[:, 1] = -points[:, 1]
        return points, label


class RandomJitter(object):
    """
    Shifts all points a random distance. The distance shifted is different for every point
    """

    def __init__(self, sigma=0.01, clip=0.05):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, points, label):
        assert (self.clip > 0)
        jitter = np.clip(self.sigma * np.random.randn(points.shape[0], 3), -1 * self.clip, self.clip)
        points += jitter
        return points, label


class RandomDropout(object):
    """

    WORKS ON BATCHES
    randomly sets points to first point in dataset, a random ratio of points is selected with max_dropout_ratio
    as upper bound. CAUTION: Do not apply before RandomJitter
    """

    def __init__(self, max_dropout_ratio=0.8):
        self.max_dropout_ratio = max_dropout_ratio

    def __call__(self, points, label):
        dropout_ratio = np.random.random() * self.max_dropout_ratio
        npoints = np.round(dropout_ratio * len(points)).astype(int)
        drop_idx = np.random.choice(len(points), npoints, replace=False)
        points[drop_idx] = np.zeros((npoints, 3)) + points[0]
        label[drop_idx] = None
        return points, label
