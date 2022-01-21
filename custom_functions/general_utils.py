import torch
import shutil
import pathlib
import sys
import importlib
import os
from sklearn.neighbors import NearestNeighbors
import transform as t


def get_device(cuda_preference=True):
    print('cuda available:', torch.cuda.is_available(),
          '; cudnn available:', torch.backends.cudnn.is_available(),
          '; num devices:', torch.cuda.device_count())

    use_cuda = False if not cuda_preference else torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    device_name = torch.cuda.get_device_name(device) if use_cuda else 'cpu'
    print('Using device', device_name)
    return device


def gen_split(percentages=(0.5, 0.2),
              paths=("/content/Pointnet_Pointnet2_pytorch/data/trainsplit.npy",
                     "/content/Pointnet_Pointnet2_pytorch/data/valsplit.npy"),
              sample_number=252,
              shuffle=True,
              seed=1):
    import random
    import numpy as np
    if shuffle:
        random.seed(seed)
        indices = range(sample_number)
        indices = np.array(random.sample(indices, sample_number))
    else:
        indices = np.arange(0, sample_number)
    start, percentage = 0, 0
    for i, (path) in enumerate(paths):
        percentage += percentages[i]
        stop = np.floor(percentage * sample_number).astype(int)
        index_subset = indices[start:stop]
        np.save(path, index_subset)
        start = stop


def get_model(source_path, device):
    model_name = set(os.listdir(source_path)) - set(
        ["pointnet2_utils.py", "logs", "checkpoints", "performance", "split", "__pycache__"])
    print(set(os.listdir(source_path)))
    model_name = list(model_name)[0]
    model_name = model_name[0:-3]
    sys.path.append(source_path)
    model = importlib.import_module(model_name)

    def inplace_relu(m):
        classname = m.__class__.__name__
        if classname.find('ReLU') != -1:
            m.inplace = True

    classifier = model.get_model(2, normal_channel=False).to(device)
    classifier.apply(inplace_relu)

    model_path = source_path + "/checkpoints/best_model.pth"
    checkpoint = torch.load(model_path)
    classifier.load_state_dict(checkpoint['model_state_dict'])

    return classifier


def gen_pred(classifier, tree_number, treedataset, device):
    # predict targets for arbitrary tree number
    points, label, target, _, upoints = treedataset[tree_number]
    points, label, target = torch.tensor(points), torch.tensor(label), torch.tensor(target)
    points, target = torch.unsqueeze(points, 0), torch.unsqueeze(target, 0)
    points, label, target = points.float().to(device), label.long().to(device), target.long().to(device)
    points = points.transpose(2, 1)

    def to_categorical(y, num_classes):
        """ 1-hot encodes a tensor """
        new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
        if (y.is_cuda):
            return new_y.cuda()
        return new_y

    with torch.no_grad():
        classifier.eval()
        result = classifier(points, to_categorical(label, 1))[0]

    preds = torch.argmax(result[0], axis=1)
    points = points[0].T
    target = target[0]
    points = points[:, :3]
    points = points.detach().cpu().numpy()
    preds = preds.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    m = torch.nn.Softmax()
    pred_probabilities = m(result[0])[:, 1].detach().cpu().numpy()

    return pred_probabilities, upoints


def find_neighbours(upoints, allpoints, k):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(upoints)
    neighbours_indices = nbrs.kneighbors(allpoints, k, return_distance=False)

    return neighbours_indices


def extrapolate(pred_probabilities, neighbours_indices):
    # produce nxk array
    mapped_probabilities = pred_probabilities[neighbours_indices]

    return np.mean(mapped_probabilities, axis=1)


def multi_sample_ensemble(source_path, npoints, tree_number, n_samples=5):
    split_path = source_path + "/split/valsplit.npy"
    root = "/content/Pointnet_Pointnet2_pytorch/data/"

    device = get_device()
    classifier = get_model(source_path, device)
    dataset = dset.PartNormalDataset(root=root,
                                     npoints=npoints,
                                     transform=t.Compose([t.Normalize()]),
                                     splitpath=split_path,
                                     normal_channel=False, mode="eval")
    # generate n_samples predictions
    allpoints = dataset[tree_number][3][0]
    targets = dataset[tree_number][3][1]
    preds = np.empty((len(allpoints), n_samples))
    for i in range(n_samples):
        pred_probabilities, upoints = gen_pred(classifier, tree_number=tree_number, treedataset=dataset, device=device)
        indices = find_neighbours(upoints, allpoints, 5)
        preds[:, i] = extrapolate(pred_probabilities, indices)

    prediction = np.mean(preds, axis=1)

    return prediction, allpoints, targets

def multi_model_ensemble(source_paths, npoints, tree_number, n_samples):

    for i in source_paths:
        