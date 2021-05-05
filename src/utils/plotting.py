from distutils.spawn import find_executable

import matplotlib.pyplot as plt
import numpy as np


def init_plot_style():
    """Initialize the plot style for pyplot.
    """
    plt.rcParams.update({'figure.figsize': (12, 9)})
    plt.rcParams.update({'lines.linewidth': 2})
    plt.rcParams.update({'lines.markersize': 12})
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



def ex2_signal_time_frequency(x,y,y_0,y_ln,f,Y,Y_0,Y_ln, fs=1,figsize=(12, 9),title=""):

    fig,ax=plt.subplots(2,figsize=(12,9))

    ax[0].plot(x, y, label='Original system',lw=7,alpha=0.5)
    ax[0].plot(x, y_0, label='Taylor approximation at c = 0',lw=3,linestyle='dashed')
    ax[0].plot(x, y_ln, label='Taylor approximation at c = ln(2)',lw=3,linestyle='dotted')
    ax[0].set_xlabel('n')
    ax[0].set_ylabel('f(x[n])')
    ax[0].legend(loc=1,prop={'size': 15})
    ax[0].set_title(title)
    ax[0].grid()

    markers, stems, _  = ax[1].stem(f,Y, label='Original Spectrum',markerfmt = 'C0o', linefmt = 'C0-',basefmt=" ")
    stems.set_linewidth(4) 
    stems.set_alpha(0.5)
    markers.set_markersize(15)
    markers.set_alpha(0.5)
    markers, _, _  = ax[1].stem(f,Y_0, label='Taylor approximation at c = 0',markerfmt = 'C1o', linefmt = 'C1--',basefmt=" ")
    markers.set_markersize(9)
    markers, _, _  = ax[1].stem(f,Y_ln, label='Taylor approximation at c = ln(2)',markerfmt = 'C2o', linefmt = 'C2:',basefmt=" ")
    markers.set_markersize(5)
    ax[1].set_xlabel('Angular frequency')
    ax[1].set_ylabel('DTFT Magnitude')
    ax[1].set_xticks([0,fs/2*2/5,fs/2*4/5,fs/2,-fs/2*2/5,-fs/2*4/5,-fs/2])
    labels = ['$0$', r'$2\pi/5$', r'$4\pi/5$', r'$\pi$', r'$-2\pi/5$', r'-$4\pi/5$', r'-$\pi$']
    ax[1].set_xticklabels(labels)
    ax[1].legend(loc=1,prop={'size': 15})
    ax[1].grid(True)

    plt.tight_layout()