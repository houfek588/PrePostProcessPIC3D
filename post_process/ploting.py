"""
ploting.py - Vykreslování 1D, 2D a 3D grafů pomocí Matplotlib pro iPIC3D postprocessing.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

plot_immediately = False


class PlotDescription:
    """Datová struktura popisující parametry a popisky grafu."""

    def __init__(self, title="title", label_x="axis x", label_y="axis y", label_z="axis z"):
        self.title = title
        self.label_x = label_x
        self.label_y = label_y
        self.label_z = label_z

        self.y_max = 0
        self.y_min = 0
        self.x_max = 0
        self.x_min = 0
        self.num_lines = 0
        self.data_labels = []

    def set_ylim(self, minimum, maximum):
        self.y_max = maximum
        self.y_min = minimum

    def set_xlim(self, minimum, maximum):
        self.x_max = maximum
        self.x_min = minimum

    def set_num_graphs(self, num_lines):
        self.num_lines = num_lines

    def multidata_labels(self, labels):
        for l in labels:
            self.data_labels.append(l)


def plot_data(dataX, dataY, descr: PlotDescription, save_to_file: bool = False, file_name="plot.png"):
    """Vykreslí 1D křivku nebo sadu křivek."""
    scale = 2
    fig = plt.figure(figsize=(16 / scale, 9 / scale))

    dataY_arr = np.asarray(dataY)
    if dataY_arr.ndim > 1 and len(dataY) < 5:
        for i in range(len(dataY)):
            label = descr.data_labels[i] if i < len(descr.data_labels) else f"Series {i}"
            plt.plot(dataX, dataY[i], label=label)
        plt.legend()
    else:
        plt.plot(dataX, dataY)

    plt.xlabel(descr.label_x)
    plt.ylabel(descr.label_y)
    plt.title(descr.title)
    plt.grid(True)

    if descr.y_max != 0 or descr.y_min != 0:
        plt.ylim(descr.y_min, descr.y_max)

    if descr.x_max != 0 or descr.x_min != 0:
        plt.xlim(descr.x_min, descr.x_max)

    if save_to_file:
        os.makedirs(os.path.dirname(os.path.abspath(file_name)), exist_ok=True)
        plt.savefig(file_name, dpi=300, bbox_inches="tight")

    if plot_immediately:
        plt.show()
    elif save_to_file:
        plt.close(fig)


def plot3D_data(dataX, dataY, dataZ, descr: PlotDescription, save_to_file: bool = False, file_name="plot.png"):
    """Vykreslí 3D povrchový graf (surface plot)."""
    scale = 1
    fig = plt.figure(figsize=(16 / scale, 9 / scale))
    ax = fig.add_subplot(111, projection="3d")

    X, Y = np.meshgrid(dataX, dataY)
    Z = np.asarray(dataZ)

    surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")
    fig.colorbar(surf, shrink=0.5, aspect=10, ax=ax)

    ax.dist = 10
    ax.set_title(descr.title)
    ax.set_xlabel(descr.label_x)
    ax.set_ylabel(descr.label_y)
    ax.set_zlabel(descr.label_z)

    if save_to_file:
        os.makedirs(os.path.dirname(os.path.abspath(file_name)), exist_ok=True)
        plt.savefig(file_name, dpi=300, bbox_inches="tight")

    if plot_immediately:
        plt.show()
    elif save_to_file:
        plt.close(fig)


def plot3Dwire_data(dataX, dataY, dataZ, descr: PlotDescription, save_to_file: bool = False, file_name="plot.png"):
    """Vykreslí 3D drátový model (wireframe)."""
    scale = 1
    fig, ax1 = plt.subplots(figsize=(16 / scale, 9 / scale), subplot_kw={"projection": "3d"})

    X, Y = np.meshgrid(dataX, dataY)
    Z = np.asarray(dataZ)
    spacing = max(1, len(Z) // 10)

    ax1.plot_wireframe(X, Y, Z, rstride=spacing, cstride=0)
    ax1.set_title(descr.title)
    ax1.set_xlabel(descr.label_x)
    ax1.set_ylabel(descr.label_y)
    ax1.set_zlabel(descr.label_z)
    plt.tight_layout()

    if save_to_file:
        os.makedirs(os.path.dirname(os.path.abspath(file_name)), exist_ok=True)
        plt.savefig(file_name, dpi=300, bbox_inches="tight")

    if plot_immediately:
        plt.show()
    elif save_to_file:
        plt.close(fig)


def plot3Dplane_data(dataX, dataY, dataZ, descr: PlotDescription, save_to_file: bool = False, file_name="plot.png", max_bar_val=None):
    """Vykreslí 2D řez pomocí imshow (časoprostorový vývoj nebo 2D FFT)."""
    scale = 1.5
    fig = plt.figure(figsize=(16 / scale, 9 / scale))
    ax = fig.add_subplot()

    Z = np.asarray(dataZ)
    x0, x1 = dataX[0], dataX[-1]
    y0, y1 = dataY[0], dataY[-1]

    im = ax.imshow(Z, aspect="auto", extent=(x0, x1, y0, y1), cmap="viridis", origin="lower", vmax=max_bar_val)
    ax.set_title(descr.title)
    ax.set_xlabel(descr.label_x)
    ax.set_ylabel(descr.label_y)

    fig.colorbar(im, ax=ax, label=descr.label_z)

    if save_to_file:
        os.makedirs(os.path.dirname(os.path.abspath(file_name)), exist_ok=True)
        plt.savefig(file_name, dpi=300, bbox_inches="tight")

    if plot_immediately:
        plt.show()
    elif save_to_file:
        plt.close(fig)


def plot_fft(dataX, dataY, descr: PlotDescription, save_to_file: bool = False, file_name="plot.png"):
    """Vypočítá 1D FFT spektrum a vykreslí jej."""
    if len(dataX) < 2:
        return

    step = float(np.diff(dataX).mean())
    if step == 0:
        return

    fft_result = np.fft.fft(dataY)
    frequencies = np.fft.fftfreq(len(dataY), d=step)
    magnitude = np.abs(fft_result)

    n_half = len(frequencies) // 2
    plot_data(frequencies[1:n_half], magnitude[1:n_half], descr, save_to_file, file_name)


def plot_histogram(dataX, levels, descr: PlotDescription, weights=None, save_to_file: bool = False, file_name="plot.png"):
    """Vykreslí histogram distribučních funkcí částic."""
    scale = 2
    fig = plt.figure(figsize=(16 / scale, 9 / scale))

    is_multi = isinstance(dataX, (list, tuple)) and len(dataX) > 0 and isinstance(dataX[0], (list, tuple, np.ndarray))

    if is_multi and len(dataX) < 5:
        for i in range(len(dataX)):
            label = descr.data_labels[i] if i < len(descr.data_labels) else f"Species {i}"
            w = weights[i] if weights is not None and i < len(weights) else None
            plt.hist(dataX[i], bins=levels, label=label, weights=w, alpha=0.7)
        plt.legend()
    else:
        plt.hist(dataX, bins=levels, weights=weights)

    plt.xlabel(descr.label_x)
    plt.ylabel(descr.label_y)
    plt.title(descr.title)
    plt.grid(True)

    if descr.y_max != 0 or descr.y_min != 0:
        plt.ylim(descr.y_min, descr.y_max)

    if descr.x_max != 0 or descr.x_min != 0:
        plt.xlim(descr.x_min, descr.x_max)

    if save_to_file:
        os.makedirs(os.path.dirname(os.path.abspath(file_name)), exist_ok=True)
        plt.savefig(file_name, dpi=300, bbox_inches="tight")

    if plot_immediately:
        plt.show()
    elif save_to_file:
        plt.close(fig)


def plot_all_graphs():
    """Zobrazí všechny vytvořené grafy na obrazovce."""
    plt.show()
