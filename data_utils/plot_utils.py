import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot(title, label, result_path, yscale='linear', legend=["Training", "Validation"],
         thinning=3):
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
    training = np.load(result_path)[:, 0]
    validation = np.load(result_path)[:, 1]
    epoch_array = np.arange(len(training)) + 1

    results = np.hstack([training.reshape(len(training), 1), validation.reshape(len(training), 1)])
    # df = pd.DataFrame(results, columns=["train_results", "val_results"])

    training = training[::thinning]
    validation = validation[::thinning]
    epoch_array = epoch_array[::thinning]




    sns.set(style='ticks')
    plt.scatter(epoch_array, training, alpha=1, s=15)
    plt.scatter(epoch_array, validation, alpha=1, s=15)

    legend = [legend[0], legend[1]]
    plt.legend(legend)
    plt.xlabel('Epoch')
    plt.ylabel(label)
    plt.yscale(yscale)
    plt.title(title)
    plt.title(title, fontsize=15)
    plt.show()
