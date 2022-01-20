import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot(title, label, result_path, yscale='linear', save_path=None, legend=["Train results", "Validation results"]):
    """Plot learning curves.

    Args:
        title (str): Title of plot
        label (str): x-axis label
        result_path: path to .npy file with results
        yscale (str, optional): Matplotlib.pyplot.yscale parameter.
            Defaults to 'linear'.
        save_path (str, optional): If passed, figure will be saved at this path.
            Defaults to None.
    """
    train_results = np.load(result_path)[:, 0]
    val_results = np.load(result_path)[:, 1]

    epoch_array = np.arange(len(train_results)) + 1
    train_label, val_label = "Training " + label.lower(), "Validation " + label.lower()

    sns.set(style='ticks')

    plt.plot(epoch_array, train_results, epoch_array, val_results, linestyle='dashed', marker='o', zorder=-1)
    legend = [legend[0], legend[1]]

    plt.legend(legend)
    plt.xlabel('Epoch')
    plt.ylabel(label)
    plt.yscale(yscale)
    plt.title(title)

    # sns.despine(trim=True, offset=5)
    plt.title(title, fontsize=15)
    if save_path:
        plt.savefig(str(save_path), bbox_inches='tight')
    plt.show()
