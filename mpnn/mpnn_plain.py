#!/usr/bin/env python

import os
import sys
import copy
import torch
import functools
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

from .utils_mpnn import kB, plot_integrand_surr, downselect, Rhombus, MPNNet, MPNNMap, WFcn, SModel, MultiModel
from .utils_nn import MLP, NNWrap, PeriodicLoss
from .utils_gen import myrc, tch, cartes_list, rel_l2, random_str, plot_dm, plot_yx

class MPNN_plain():
    """docstring for MPNN"""
    def __init__(self, case_glob):
        self.pnames = case_glob['pnames']
        self.mpts = case_glob['mpts']
        self.ncl = len(self.mpts)
        self.ndim = len(self.pnames)
        #self.centered = True
        self.debug = True

        self.mmodel = None # set in eval()
        self.expon = True # this means e^NN

        #self.set_rhombi()


    def __repr__(self):
        return f"MPNN Object with {self.ncl} minima"

    def printInfo(self):
        print(f"Number of minima: {self.ncl}")


    def plot_xdata(self, xall):

        for idim in range(self.ndim):
            for jdim in range(idim+1, self.ndim):
                plt.figure(figsize=(10,10))
                plt.plot(xall[:, idim], xall[:, jdim], 'o', alpha=0.5, markeredgecolor='w')
                for mpt in self.mpts:
                    plt.plot(mpt.center[idim], mpt.center[jdim], 'o', markersize=16, zorder=10000)

                plt.xlabel(f'{self.pnames[idim]}')
                plt.ylabel(f'{self.pnames[jdim]}')
                plt.savefig(f'xtrain_{idim}_{jdim}.png')
                plt.clf()


    def fit(self, xall, yall,
            tr_frac=0.9, cl_rad=1.e+10,
            hls=(111,111, 111), activ='relu',
            lr=0.001, nepochs=2000, bsize=1000,
            eps=0.02):
        myrc()
        self.plot_xdata(xall)


        nall, dim = xall.shape
        assert(yall.shape[0] == nall)
        assert(dim == self.ndim)

        indperm = np.random.permutation(range(nall))
        ntrn = int(tr_frac*nall) #7000 #int(sys.argv[1]) #9380
        ntst = nall - ntrn #int(sys.argv[2]) #nall - ntrn

        indtrn = indperm[:ntrn]
        indtst = indperm[-ntst:]

        np.savetxt('indtrn.txt', indtrn, fmt='%d')
        np.savetxt('indtst.txt', indtst, fmt='%d')


        for i in range(self.ncl):
            for j in range(i+1, self.ncl):
                dd = np.linalg.norm(self.mpts[i].center - self.mpts[j].center)
                print(f"dist(c_{i}, c_{j})={dd}")


        def lrfcn(epoch):
            if epoch < 5000:
                return lr
            else:
                return lr/3.

        self.ptmodels = []
        for j in range(self.ncl):

            dists = cdist(xall, np.array([pt.center for pt in self.mpts]))


            xtrn, xtst = xall[indtrn, :], xall[indtst, :]
            ytrn, ytst = yall[indtrn], yall[indtst]
            dists_trn, dists_tst = dists[indtrn, :], dists[indtst, :]

            print("======================================")
            indsel = downselect(dists_trn[:, j], 1.e-15, cl_rad, ytrn, self.mpts[j].yshift)
            print("Training size of this cluster : ", indsel.shape[0])
            np.savetxt(f'indsel_{j}_trn.txt', indsel, fmt='%d')
            xtrn_this, ytrn_this = xtrn[indsel, :], ytrn[indsel]
            print("Min ", np.min(ytrn_this-self.mpts[j].yshift))

            indsel = downselect(dists_tst[:, j], 1.e-15, cl_rad, ytst, self.mpts[j].yshift)
            print("Testing size of this cluster : ", indsel.shape[0])
            np.savetxt(f'indsel_{j}_tst.txt', indsel, fmt='%d')
            xtst_this, ytst_this = xtst[indsel, :], ytst[indsel]

            mapparams = [self.mpts[j].center, self.mpts[j].hess, self.mpts[j].yshift]
            mpnn_map = MPNNMap(mapparams)

            ptmodel, ytrn_pred, ytst_pred = fit(xtrn_this, ytrn_this, mpnn_map,
                                                hls = hls, activ=activ,
                                                nepochs=nepochs, bsize=bsize,
                                                lr=1.0, lmbd=lrfcn,
                                                xtst=xtst_this, ytst=ytst_this,
                                                debug=self.debug)


            self.ptmodels.append(ptmodel)


        ytrn_pred = self.eval(xall[indtrn, :], eps=eps)

        print("=========================================")
        print("MPNN Training error: ", rel_l2(yall[indtrn], ytrn_pred))
        ytst_pred = self.eval(xall[indtst, :], eps=eps)
        mpnn_tst_err = rel_l2(yall[indtst], ytst_pred)
        print("MPNN Testing error: ", mpnn_tst_err)
        # print(np.min(ytrn_pred), np.min(ytst_pred), np.min(ytrn), np.min(ytst))
        # sys.exit()
        plot_integrand_surr(xall[indtrn, :], yall[indtrn], ytrn_pred, xall[indtst, :], yall[indtst], ytst_pred)

        # For debugging
        if self.debug:
            np.savetxt('ytrn_all', yall[indtrn])
            np.savetxt('ytrn_pred_all', ytrn_pred)
            np.savetxt('ytst_all', yall[indtst])
            np.savetxt('ytst_pred_all', ytst_pred)



        plot_dm([yall[indtrn], yall[indtst]], [ytrn_pred, ytst_pred], errorbars=[], labels=['Training', 'Testing'],
                    axes_labels=['Model', 'Apprx'], figname='dm.png',
                    showplot=False, legendpos='in', msize=4)
        plot_dm([np.log(yall[indtrn]), np.log(yall[indtst])], [np.log(ytrn_pred), np.log(ytst_pred)], errorbars=[], labels=['Training', 'Testing'],
                    axes_labels=['E', 'E_s'], figname='dm_log.png',
                    showplot=False, legendpos='in', msize=4)

        return mpnn_tst_err

    def eval(self, xx, eps=0.02, temp=None, krnl='exp'):

        assert(self.ptmodels is not None)
        assert(self.mpts is not None)
        if (len(xx.shape) == 1):
            xx = xx.reshape(1, -1)

        wfcn = WFcn(self.mpts, eps, rhomb=None, krnl=krnl)

        models = [SModel(ptmodel, mpt, None, expon=self.expon) for ptmodel, mpt in zip(self.ptmodels, self.mpts)]
        self.mmodel = MultiModel(models, wfcn=wfcn)


        yy = self.mmodel(xx)
        # print(np.min(yy))
        if temp is not None:
            yy = np.exp(-yy / (kB * temp))

        return yy

##########################################################################################
##########################################################################################
##########################################################################################


def fit(xtrn, ytrn, xymap, hls=(111, 111, 111), activ='relu', nepochs=5000, bsize=None, lr=0.01, lmbd=None, xtst=None, ytst=None, debug=True):
    print(f"Number of training points {ytrn.shape[0]}")
    print(f"Number of testing points {ytst.shape[0]}")
    xtrn_, ytrn_ = xymap.fwd(xtrn, ytrn)
    if ytst is not None:
        xtst_, ytst_ = xymap.fwd(xtst, ytst)

    if bsize is None:
        bsize = xtrn_.shape[0]

    mlp = MLP(xtrn_.shape[1], 1, hls, biasorno=True,
                 activ=activ, bnorm=False, bnlearn=True, dropout=0.0,
                 final_transform=None)


    model = mlp.fit(xtrn_, ytrn_.reshape(-1,1), val=[xtst_, ytst_.reshape(-1,1)],
                     lrate=lr, lmbd=lmbd, batch_size=bsize, nepochs=nepochs,
                     gradcheck=False, freq_out=100, freq_plot=100)

    Surr = NNWrap(model)


    if ytst is not None:
        ytst_pred_ = Surr(xtst_).reshape(-1,)
        yerr_tst_ = rel_l2(ytst_pred_, ytst_)
        print(f"Rel-L2 NN Error at testing points {yerr_tst_}")
        xtst_copy, ytst_pred = xymap.inv(xtst_, ytst_pred_)

        xerr = rel_l2(xtst_copy, xtst)
        assert(xerr < 1.e-16)
        yerr_tst = rel_l2(ytst_pred, ytst)
        print(f"Rel-L2 E Error at testing points {yerr_tst}")

        # For debugging
        if debug:
            np.savetxt('ytst.dbg', ytst)
            np.savetxt('ytst_pred_.dbg', ytst_pred_)
            np.savetxt('ytst_pred.dbg', ytst_pred)
            np.savetxt('xtst.dbg', xtst)
            np.savetxt('ytst_.dbg', ytst_)
            np.savetxt('xtst_.dbg', xtst_)

    else:
        ytst_pred = None

    ytrn_pred_ = Surr(xtrn_).reshape(-1,)
    yerr_trn_ = rel_l2(ytrn_pred_, ytrn_)
    print(f"Rel-L2 NN Error at training points {yerr_trn_}")

    xtrn_copy, ytrn_pred = xymap.inv(xtrn_, ytrn_pred_)
    xerr = rel_l2(xtrn_copy, xtrn)
    assert(xerr < 1.e-16)
    yerr_trn = rel_l2(ytrn_pred, ytrn)
    print(f"Rel-L2 E Error at training points {yerr_trn}")

    # For debugging
    if debug:
        np.savetxt('ytrn.dbg', ytrn)
        np.savetxt('ytrn_pred_.dbg', ytrn_pred_)
        np.savetxt('ytrn_pred.dbg', ytrn_pred)
        np.savetxt('xtrn.dbg', xtrn)
        np.savetxt('ytrn_.dbg', ytrn_)
        np.savetxt('xtrn_.dbg', xtrn_)
        rs = random_str(4)
        os.system(f'mkdir {rs}; mv *.dbg loss_history.png loss_history_log.png {rs}/')

    return model, ytrn_pred, ytst_pred
