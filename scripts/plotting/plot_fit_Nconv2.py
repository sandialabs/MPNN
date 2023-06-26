#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


####################################################################################
def myrc():
    mpl.rc('legend', loc='best', fontsize=22)
    mpl.rc('lines', linewidth=4, color='r')
    mpl.rc('axes', linewidth=3, grid=True, labelsize=22)
    mpl.rc('xtick', labelsize=20)
    mpl.rc('ytick', labelsize=20)
    mpl.rc('font', size=20)
    mpl.rc('figure', figsize=(12, 9), max_open_warning=200)
    # mpl.rc('font', family='serif')

    return mpl.rcParams

####################################################################################
####################################################################################

nrepl = 40
NNs = [125, 250, 500, 1000, 2000, 4000, 8000]
TTs = [400, 600, 800, 1000, 1200, 1400]

nnum = len(NNs)
nt = len(TTs)

# rrmse_all = np.zeros((nt, nnum, nrepl, 4))
# for inum, N in enumerate(NNs):
#     for ir in range(nrepl):
#         rrmse_all[:, inum, ir, 0] = N
#         rrmse_all[:, inum, ir, 1] = np.array(TTs)
#         rrmse_all[:, inum, ir, 2] = ir+1
#         rrmse_all[:, inum, ir, 3] = np.loadtxt(f'../r{ir+1}/rrmse_test_N{N}.txt')[:, -1]

# np.savetxt(f'rrmse_all.txt', rrmse.reshape(-1,4))
# for it, T in enumerate(TTs):
#     rr = rrmse[it, :, :, :].reshape(-1,4)
#     np.savetxt(f'rrmse_T{int(T)}.txt', rr)
# print(rrmse)
# sys.exit()

####################################################################################
####################################################################################

rrmse_all = np.loadtxt('rrmse_all.txt').reshape((nt, nnum, nrepl, 4))

myrc()

fig = plt.figure(figsize=(12,9))

for it, T in enumerate(TTs):
    rrmse_T  = rrmse_all[it,:,:, -1]
    #rrmse_T = np.loadtxt(f'rrmse_T{int(T)}.txt')[:,-1].reshape(nnum, nrepl)
    errs_qt = np.quantile(rrmse_T, [0.25,0.5,0.75], axis=1)
    #rrmse = np.mean(rrmse_T, axis=2)[it,:]
    rrmse = errs_qt[1,:]
    rrmse_lower = errs_qt[0, :]
    rrmse_upper = errs_qt[2, :]

    if it % 2 == 0: # only plot every other temp
        plt.plot(NNs, rrmse, 'o-', label='T = '+str(T))
        #plt.errorbar(NNs, rrmse, yerr=[rrmse-rrmse_lower,rrmse_upper-rrmse],fmt='o-', label='T = '+str(T))

xticklabels = [int(ns) for ns in NNs]
plt.xscale('log')
#plt.yscale('log')
#plt.ylim(0.0, 0.31)
plt.xticks(NNs, xticklabels, rotation=80)
# yticks = [0.0, 0.05, 0.1, 0.15, 0.2]
# plt.gca().set_yticks(yticks)
#plt.gca().set_yticklabels(["$%.1f$" % y for y in yticks], fontsize=18);
plt.legend()
plt.xlabel("Number of Training Samples, N")
plt.ylabel("Relative RMSE for  $e^{-V/kT}$  Surrogate")
fig.tight_layout()
plt.savefig('rrmse.png')
