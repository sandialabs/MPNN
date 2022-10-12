#!/usr/bin/env python

import os
import numpy as np
import matplotlib.pyplot as plt

from mpnn.utils_mpnn import kB
from mpnn.utils_gen import myrc, read_textlist

myrc()

ytrain = np.loadtxt('ytrain.txt')
ltrain = read_textlist('ltrain.txt', len(ytrain))

T=400
beta = 1./(kB*T)
ms = 8

labels_unique = list(set(ltrain))
labels_unique.sort()
print(labels_unique)
n_groups = len(labels_unique)
styles = ['mo', \
          'ro', 'rd', 'rx', \
          'bd', 'bd', 'bd', 'bd', 'bd', 'bd', 'bd', 'bd', \
          'go', 'gd', 'gx', 'gs', 'gs', \
          'bo', 'bd', 'bx', 'bs', 'bs' \
          ] # this is optimized for the CO-on-Pt(111) case
styles = ['ro', 'rd', 'rs', \
          'bo', 'bd', 'bs', \
          'mx', 'gx'
          ] # this is optimized for the CH3-on-Ni(111) case

for i in range(n_groups)[::-1]:
    yy = ytrain[labels_unique[i]==np.array(ltrain)]
    print(f"Group {i+1}/{n_groups} w {yy.shape[0]} points")
    plt.plot(yy, np.exp(-beta*yy), styles[i],
             markeredgecolor='grey', markersize=ms, label=labels_unique[i])



plt.xlabel(r'$E$', fontsize=20)
plt.ylabel(r'$e^{-E/kT}$   for   $T = '+str(T)+'$', fontsize=20)
plt.legend(fontsize=10)

plt.savefig(f'E_vs_eE_T{T}.png')

plt.xscale('log')
plt.savefig(f'E_vs_eE_log_T{T}.png')
