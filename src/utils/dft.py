import numpy as np
import scipy.fft as sf
import scipy.signal as ss


def dft(x, fs, n_dft=None, window='hann', onesided=True):
    """ Computes the DFT of a given signal.

    Returns the DFT of the input signal and a corresponding frequency vector.

    Parameters
    ----------
    x : numpy.ndarray
        The 1D input signal.
    fs : float
        The sampling frequency used to obtain the input signal.
    n_dft : int
        Number of points the DFT should compute. If 'None' this defaults to the number of samples in the input signal.
    window : str
        The type of window to apply. Admissible window types are compatible with scipy.signal.get_window().
    onesided : bool
        If true, returns the one-sided spectrum only. Returns the two-sided spectrum otherwise.

    Returns
    ------
    (numpy.ndarray, numpy.ndarray)
        The input signal's DFT and a corresponding frequency vector.
    """
    if n_dft is None:
        n_dft = len(x)

    # generate and plot window
    win = ss.get_window(window, len(x))
    win /= np.sum(win)  # normalize window

    # this is a little hack to avoid leakage around f=0 which would mess up our scaling in the onesided case
    xmean = np.mean(x)
    x -= xmean

    # compute the DFT and generate the corresponding frequency vector
    xdft = np.abs(sf.fft(x * win, n_dft))
    xdft[0] = xmean  # manually add the signal mean (i.e. amplitude @ f=0) bec we removed it before computing the fft
    freq = np.arange(n_dft) / n_dft * fs

    if onesided:
        # take only the one-sided spectrum
        num_onesided = np.int(n_dft / 2) + (n_dft % 2)  # number of DFT samples in the one-sided spectrum
        freq = freq[:num_onesided]
        xdft = xdft[:num_onesided]
        xdft[1:] *= 2  # correct scaling
    else:
        # get the correct two-sided spectrum
        xdft = sf.fftshift(xdft)

    return xdft, freq
