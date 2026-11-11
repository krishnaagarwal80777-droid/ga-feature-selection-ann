import matplotlib.pyplot as plt
import numpy as np


def plot_feature_frequency(selections, attribute_names, output_path):
    selections = np.asarray(selections, dtype=bool)
    attribute_names = np.asarray(attribute_names)
    if selections.ndim != 2 or selections.shape[1] != len(attribute_names):
        raise ValueError("selection columns must match the attribute names")

    frequencies = selections.mean(axis=0)
    order = np.argsort(frequencies)
    fig, ax = plt.subplots(figsize=(9, max(4, len(attribute_names) * 0.25)))
    ax.barh(attribute_names[order], frequencies[order])
    ax.set_xlabel("selection frequency")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)