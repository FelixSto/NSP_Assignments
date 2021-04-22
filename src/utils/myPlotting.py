from distutils.spawn import find_executable

import matplotlib.pyplot as plt
import numpy as np


def ex2_signal_time_frequency(x,y,y_0,y_ln,f,Y,Y_0,Y_ln, fs=1,figsize=(12, 9)):

    fig,ax=plt.subplots(2,figsize=(12,9))

    ax[0].plot(x, y, label='Original system',lw=7,alpha=0.5)
    ax[0].plot(x, y_0, label='Taylor approximation at c = 0',lw=3,linestyle='dashed')
    ax[0].plot(x, y_ln, label='Taylor approximation at c = ln(2)',lw=3,linestyle='dotted')
    ax[0].set_xlabel('n')
    ax[0].set_ylabel('f(x[n])')
    ax[0].legend(loc=1,prop={'size': 15})
    ax[0].grid()

    markers, stems, _  = ax[1].stem(f,Y, label='Input Spectrum',markerfmt = 'C0o', linefmt = 'C0-',basefmt=" ")
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