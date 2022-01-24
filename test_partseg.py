import os
import torch
import sys
import importlib
import shutil
import numpy as np
import custom_functions.transform as t
from tqdm import tqdm
import custom_functions.general_utils as gu
sys.path.append("/content/Pointnet_Pointnet2_pytorch/data_utils")


def evaluate_model(npoints, source_path, ensemble="sample", method="mean", n_samples=5):
    if ensemble == "model":
        split_path = source_path[0] + "/split/valsplit.npy"
    else:
        split_path = source_path + "/split/valsplit.npy"

    valindices = np.load(split_path)
    f1score, precision, recall, total, correct = [], [], [], [], []

    for i in tqdm(range(len(valindices))):
        if ensemble == "sample":
            pred, allpoints, target, best_thresholds = gu.multi_sample_ensemble(source_path, npoints, tree_number=i, n_samples=n_samples, method=method)
        elif ensemble == "model":
            # source path needs to be a list for this
            pred, allpoints, target, best_thresholds = gu.multi_model_ensemble(source_path, npoints, tree_number=i, n_samples=n_samples, method=method)

        pred_choice = (pred > 0).astype("int")

        # measures
        tp = np.sum(np.logical_and(target == 1, pred_choice == 1))
        fn = np.sum(np.logical_and(target == 1, pred_choice == 0))
        fp = np.sum(np.logical_and(target == 0, pred_choice == 1))
        tn = np.sum(np.logical_and(target == 0, pred_choice == 0))

        precision.append(tp / (tp + fp))
        recall.append(tp / (tp + fn))
        f1score.append(2 * (precision[-1] * recall[-1]) / (precision[-1] + recall[-1]))
        correct.append(tp + tn)
        total.append(len(pred))

    acc = np.array(correct) / np.array(total)
    print("")
    print("Acc:", np.sum(correct) / np.sum(total), "F1 score", np.mean(f1score), "Precision:", np.mean(precision),
          "Recall:", np.mean(recall))

    return np.array(f1score), np.array(precision), np.array(recall), acc, best_thresholds
