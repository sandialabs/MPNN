#!/usr/bin/env python

import os
import sys
import copy
import torch
import functools
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

from .utils_mpnn import kB, plot_integrand_surr, downselect, Rhombus, MPNNet, MPNNMap, WFcn, SModel, MultiModel, multiply_traindata
from .utils_nn import MLP, NNWrap, PeriodicLoss
from .utils_gen import myrc, tch, cartes_list, rel_l2, random_str, plot_dm, plot_yx

class MPNN():
    """docstring for MPNN"""
    def __init__(self, case_glob):
        self.pnames = case_glob['pnames']
        self.zmin, self.zmax, self.delta_x = case_glob['zmin'], case_glob['zmax'], case_glob['delta_x']
        self.mpts = case_glob['mpts']
        self.ncl = len(self.mpts)
        self.ndim = len(self.pnames)
        self.centered = True
        self.debug = True
        self.cushion_data = False

        self.rhombi = None # set is set_rhombi()
        self.rhomb_eval = None # set in set_eval_rhombus()
        self.mmodel = None # set in eval()
        self.eval_type = None

        self.set_rhombi()


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

    def set_rhombi(self):
        self.rhombi = []
        for j in range(self.ncl):

            init = np.array([0.,0., self.zmin])
            if self.centered:
                cc = np.array([3.* self.delta_x / 4., np.sqrt(3.)* self.delta_x / 4.,(self.zmin+self.zmax)/2.])
                init += (self.mpts[j].center[:3]-cc)

            rhomb = Rhombus(init=init,
                        delta_x=self.delta_x,
                        delta_z=self.zmax-self.zmin)
            xyfold = True
            # TODO: For MB and toy cases, need to rethink
            # rhomb = Rhombus(init=np.array([0.,0., 0.]),
            #             delta_x=1.0,
            #             delta_z=1.0)
            # xyfold = False # TODO be careful! evaluatio nin SModel always folds!
            self.rhombi.append(rhomb)

    def set_eval_rhombus(self, eval_type=-2):
        init = np.array([0.,0., self.zmin])
        self.rhomb_eval = Rhombus(init=init,
                        delta_x=self.delta_x,
                        delta_z=self.zmax-self.zmin)
        if eval_type >= 0:
            self.rhomb_eval.centerInit(self.mpts[eval_type].center)
        elif eval_type == -1:
            pass
        elif eval_type == -2:
            center_com = np.mean(np.array([self.mpts[j].center for j in range(self.ncl)]), axis=0)
            self.rhomb_eval.centerInit(center_com)

    def fit(self, xall, yall,
            tr_frac=0.9, cl_rad=1.e+10,
            hls=(111,111, 111), activ='relu',
            lr=0.001, nepochs=2000, bsize=1000, periodic_lambda=0.0,
            eps=0.02, eval_type=-2):
        myrc()
        self.eval_type = eval_type
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


            xall_cube = xall.copy()
            xall_cube[:,:3] = self.rhombi[j].toCube(xall[:,:3], xyfold=True)
            mpts_cube = self.rhombi[j].mpts_toCube(self.mpts, xyfold=True)#watch out for toy cases no folding?!



            dists = cdist(xall_cube, np.array([pt.center for pt in mpts_cube]))


            xtrn, xtst = xall_cube[indtrn, :], xall_cube[indtst, :]
            ytrn, ytst = yall[indtrn], yall[indtst]
            dists_trn, dists_tst = dists[indtrn, :], dists[indtst, :]

            print("======================================")
            indsel = downselect(dists_trn[:, j], 1.e-15, cl_rad, ytrn, mpts_cube[j].yshift)
            print("Training size of this cluster : ", indsel.shape[0])
            np.savetxt(f'indsel_{j}_trn.txt', indsel, fmt='%d')
            xtrn_this, ytrn_this = xtrn[indsel, :], ytrn[indsel]
            print("Min ", np.min(ytrn_this-self.mpts[j].yshift))

            indsel = downselect(dists_tst[:, j], 1.e-15, cl_rad, ytst, mpts_cube[j].yshift)
            print("Testing size of this cluster : ", indsel.shape[0])
            np.savetxt(f'indsel_{j}_tst.txt', indsel, fmt='%d')
            xtst_this, ytst_this = xtst[indsel, :], ytst[indsel]

            mapparams = [mpts_cube[j].center, mpts_cube[j].hess, mpts_cube[j].yshift]
            mpnn_map = MPNNMap(mapparams)

            if self.cushion_data:
                xtrn_this, ytrn_this = multiply_traindata(xtrn_this, ytrn_this, kx=1, ky=1)
            ptmodel, ytrn_pred, ytst_pred = fit(xtrn_this, ytrn_this, mpnn_map,
                                                hls = hls, activ=activ,
                                                nepochs=nepochs, bsize=bsize,
                                                lr=1.0, lmbd=lrfcn,
                                                periodic_lambda=periodic_lambda,
                                                xtst=xtst_this, ytst=ytst_this,
                                                debug=self.debug)


            self.ptmodels.append(ptmodel)


        ytrn_pred = self.eval(xall[indtrn, :], eps=eps, eval_type=eval_type)

        print("=========================================")
        print("MPNN Training error: ", rel_l2(yall[indtrn], ytrn_pred))
        ytst_pred = self.eval(xall[indtst, :], eps=eps, eval_type=eval_type)
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

    def eval(self, xx, eps=0.02, expon=True, temp=None, eval_type=None):

        assert(self.ptmodels is not None)
        assert(self.mpts is not None)
        assert(self.rhombi is not None)
        if (len(xx.shape) == 1):
            xx = xx.reshape(1, -1)

        if eval_type is None:
            eval_type = self.eval_type

        self.set_eval_rhombus(eval_type=eval_type)

        # TODO: make a function inside WFcn() for shifting rhombus, so that we can center at x and not use the original rhombus here? Upd: Not worth it, since WFcn takes multiple x's as argument!!!
        wfcn = WFcn(self.mpts, eps, rhomb=self.rhomb_eval)
        models = [SModel(ptmodel, mpt, rhomb, expon=expon) for ptmodel, mpt, rhomb in zip(self.ptmodels, self.mpts, self.rhombi)]

        mmodel = MultiModel(models, wfcn=wfcn)
        # TODO: move these to init somewhere?
        self.mmodel = mmodel

        yy = mmodel(xx)

        if temp is not None:
            yy = np.exp(-yy / (kB * temp))

        return yy

##########################################################################################
##########################################################################################
##########################################################################################


def fit(xtrn, ytrn, xymap, hls=(111, 111, 111), activ='relu', nepochs=5000, bsize=None, lr=0.01, lmbd=None, periodic_lambda=0.0, xtst=None, ytst=None, debug=True):
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

    ndim = xtrn_.shape[1]
    center, hess, yshift = xymap.center, xymap.hess, xymap.yshift
    mpnet = MPNNet(mlp, ndim, tch(center), tch(hess), tch(yshift))
    ngr = 4
    # bdry1x = np.array(cartes_list([[0.0]]+ [np.linspace(0, 1, 111)] + [[0.5]] + [[0.0]]*(ndim-3)))
    # bdry2x = np.array(cartes_list([[1.0]]+ [np.linspace(0, 1, 111)] + [[0.5]] + [[0.0]]*(ndim-3)))
    # bdry1y = np.array(cartes_list([np.linspace(0, 1, 111)] + [[0.0]] + [[0.5]] + [[0.0]]*(ndim-3)))
    # bdry2y = np.array(cartes_list([np.linspace(0, 1, 111)] + [[1.0]] + [[0.5]] + [[0.0]]*(ndim-3)))

    bdry1x = np.array(cartes_list([[0.0]]+ [np.linspace(0, 1, 11)] + [np.linspace(0, 1, ngr)]*(ndim-2)))
    bdry2x = np.array(cartes_list([[1.0]]+ [np.linspace(0, 1, 11)] + [np.linspace(0, 1, ngr)]*(ndim-2)))
    bdry1y = np.array(cartes_list([np.linspace(0, 1, 11)] + [[0.0]] + [np.linspace(0, 1, ngr)]*(ndim-2)))
    bdry2y = np.array(cartes_list([np.linspace(0, 1, 11)] + [[1.0]] + [np.linspace(0, 1, ngr)]*(ndim-2)))

    # bdry1 = np.array(cartes_list([[0.0]]+ [np.linspace(0, 1, ngr)]*(ndim-1)))
    # bdry2 = np.array(cartes_list([[1.0]]+ [np.linspace(0, 1, ngr)]*(ndim-1)))
    bdry1 = np.vstack((bdry1x, bdry1y))
    bdry2 = np.vstack((bdry2x, bdry2y))

    loss = PeriodicLoss([mpnet, periodic_lambda, tch(bdry1), tch(bdry2)])
    model = mlp.fit(xtrn_, ytrn_.reshape(-1,1), val=[xtst_, ytst_.reshape(-1,1)],
                     lrate=lr, loss=loss, lmbd=lmbd, batch_size=bsize, nepochs=nepochs,
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
