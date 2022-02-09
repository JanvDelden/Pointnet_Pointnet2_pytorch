import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def plot(title, label, result_path, yscale='linear', legend=["Training", "Validation"],
         thinning=3, save_path=None, size=(5,4)):

    training = np.load(result_path)[:, 0]
    validation = np.load(result_path)[:, 1]
    epoch_array = np.arange(len(training)) + 1

    training = training[::thinning]
    validation = validation[::thinning]
    epoch_array = epoch_array[::thinning]
    plt.figure(figsize=size)

    sns.set(style='ticks')
    plt.plot(epoch_array, training,linestyle='dashed', marker='o', zorder=-1)
    plt.plot(epoch_array, validation, linestyle='dashed', marker='o', zorder=-1)
    legend = [legend[0], legend[1]]
    plt.legend(legend)
    plt.xlabel('Epoch')
    plt.ylabel(label)
    plt.yscale(yscale)
    plt.title(title, fontsize=15)

    if save_path:
        plt.savefig(str(save_path), bbox_inches='tight', format="svg", transparent=True)
    plt.show()