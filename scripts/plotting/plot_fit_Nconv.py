#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from klpc.utils.plotting import myrc, lighten_color

myrc()

#8000 4000 2000 1000 500 250 125
nc = 7
NN_all = [125*2**i for i in range(nc)]
nt = 6
TT_all = [400+i*200 for i in range(nt)]

nrepl=40
rrmse_full = np.zeros((nt, nc, nrepl))
for ir in range(nrepl):
    for ic, NN in enumerate(NN_all):
        rrmse_full[:, ic, ir] = np.loadtxt(f'r{ir+1}/rrmse_test_N{NN}.txt')[:,2]

colors = ['r', 'g', 'b', 'm', 'orange', 'y', 'grey']

rrmse_qt = np.quantile(rrmse_full, [0.25, 0.5, 0.75], axis=2)
print(rrmse_qt.shape)


plt.figure(figsize=(12,10))

for it, TT in enumerate(TT_all[::2]):
    plt.plot(NN_all,rrmse_qt[1, it, :], 'o-', color=colors[it], ms=6, label=f'T={int(TT)}')
    #plt.fill_between(NN_all, rrmse_qt[0, it, :], rrmse_qt[2, it, :],
    #                 color=lighten_color(colors[it], 0.5), alpha=0.5)

plt.xlabel(r'Number of training samples, $N$')
plt.ylabel(r'Relative RMSE')
plt.legend(loc='upper right')
plt.savefig(f'intg_conv.png')
plt.yscale('log')
plt.gca().get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
plt.grid(True, which='minor', linestyle='--', axis='y')
plt.grid(True, which='major', linestyle='--')
plt.savefig(f'intg_conv_ylog.png')
plt.xscale('log')
plt.gca().get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
plt.gca().set_xticks(NN_all)
plt.savefig(f'intg_conv_xylog.png')
plt.yscale('linear')
plt.savefig(f'intg_conv_xlog.png')



