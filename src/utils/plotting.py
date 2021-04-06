from distutils.spawn import find_executable

import matplotlib.pyplot as plt
import numpy as np


def init_plot_style():
    """Initialize the plot style for pyplot.
    """
    plt.rcParams.update({'figure.figsize': (12, 9)})
    plt.rcParams.update({'lines.linewidth': 2})
    plt.rcParams.update({'lines.markersize': 8})
    plt.rcParams.update({'lines.markeredgewidth': 2})
    plt.rcParams.update({'axes.labelpad': 10})
    plt.rcParams.update({'xtick.major.width': 2.5})
    plt.rcParams.update({'xtick.major.size': 10})
    plt.rcParams.update({'xtick.minor.size': 5})
    plt.rcParams.update({'ytick.major.width': 2.5})
    plt.rcParams.update({'ytick.major.size': 10})
    plt.rcParams.update({'ytick.minor.size': 5})

    # for font settings see also https://stackoverflow.com/questions/2537868/sans-serif-math-with-latex-in-matplotlib
    plt.rcParams.update({'font.size': 20})
    plt.rcParams.update({'font.family': 'sans-serif'})

    # this checks if the necessary executables for rendering latex are included in your path; see also
    # https://matplotlib.org/stable/tutorials/text/usetex.html
    if find_executable('latex') and find_executable('dvipng') and find_executable('ghostscript'):
        plt.rcParams.update({'text.usetex': True})
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath,amssymb,amsfonts,amsthm}' + \
                                              r'\usepackage{siunitx}' + \
                                              r'\sisetup{detect-all}' + \
                                              r'\usepackage{helvet}' + \
                                              r'\usepackage{sansmath}' + \
                                              r'\sansmath'


def show_grayscale_img(img: np.array, figsize=(12, 9)):
    """Shows a grayscale image given as numpy array.

    Parameters
    ----------
    img : numpy.ndarray
        The image.
    figsize : tuple
        The size of the figure. Default size is (12,9).
    """
    plt.figure(figsize=figsize)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255, origin='lower')
    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
