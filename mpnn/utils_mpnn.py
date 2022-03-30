#!/usr/bin/env python

import sys
import copy
import torch
import functools
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

from .utils_gen import rel_l2, linear_transform
from .utils_nn import MLPBase

## GLOBAL variables. Not elegant but will do.
kB = 8.6173e-5  # eV/K  # OR 1.380649e-23 m2kg/s2/K
a_to_m = 1.6605402e-27
#m = um * a_to_m # CO = 4.6511705e-26, H = 1.6735575e-27 # kg
h = 6.62607004 * 1.e-34  # m2 kg / s # (Planck's constant in J*s)
ev_to_j = 1.60218e-19  # J/eV


def trans_kinetic_factor(TT, um):
    m = um * a_to_m
    return 10**(-30) * (np.sqrt(2.0 * np.pi * m * kB * TT * ev_to_j / (h**2.0)))**3.0

def rot_kinetic_factor(TT, um, momIn, ads=None):
    m = um * a_to_m

    if ads=='CO':
        return 2.0 * np.pi * np.sqrt(2.0 * np.pi * momIn * kB * TT * ev_to_j) *\
           2.0 * np.sqrt(2.0 * np.pi * momIn * kB * TT * ev_to_j) / (h**2)
    else:
        print(f"Need to implement rotational kinetic factor for {ads}. Exiting.")
        sys.exit()




def downselect(dists, rad_min, rad_max, yy, ymin):
    nn = yy.shape[0]
    indsel1 = np.arange(nn)[dists<rad_max]
    indsel2 = np.arange(nn)[dists>rad_min]
    indsel3 = np.arange(nn)[yy>ymin]

    indsel = functools.reduce(np.intersect1d, [indsel1, indsel2, indsel3])
    return indsel

class Rhombus():
    def __init__(self, init, delta_x, delta_z, verbose=False):
        self.setInit(init)  # coordinate of lower left corner
        self.delta_x = delta_x
        self.delta_z = delta_z

        self.delta_y = delta_x * np.sqrt(3.) / 2   # 6.576879959372833 /3.

        # Transform matrix from cube to rhombus
        self.transform = np.diag(np.array([self.delta_x, self.delta_y, self.delta_z]))
        self.transform[1, 0] = self.delta_x / 2.
        self.inv_transform = np.linalg.inv(self.transform)


        if verbose:
            self.printInfo()

        return

    def printInfo(self):
        print(f"Initial point: {self.init}")
        print("Transform matrix:")
        print(self.transform)
        print(f"Rhombus width(x) {self.delta_x}")
        print(f"Rhombus height(y) {self.delta_y}")
        print("Inv-Transform matrix:")
        print(self.inv_transform)

    def setInit(self, init):
        self.init = init

    def centerInit(self, x):
        cc = np.array([3.* self.delta_x / 4., np.sqrt(3.)* self.delta_x / 4.,self.delta_z/2.])
        self.init = (x[:3]-cc)


    def toCube(self, xyz, xyfold=False):

        xyz_cube = linear_transform(xyz, inshift=self.init, matrix=self.inv_transform)
        if xyfold:
            for j in range(2):
                xyz_cube[:, j] -= np.floor(xyz_cube[:, j])

        return xyz_cube

    def fromCube(self, xyz):
        return linear_transform(xyz, matrix=self.transform, outshift=self.init)

    def mpts_toCube(self, mpts, xyfold=False):
        mpts_new = []
        for mpt in mpts:
            center_new = mpt.center.copy()
            center_new[:3] = linear_transform(mpt.center[:3].reshape(1, -1), inshift=self.init, matrix=self.inv_transform).reshape(-1,)
            if xyfold:
                for j in range(2):
                    center_new[j] -= np.floor(center_new[j])


            ndim = center_new.shape[0]
            transform_all = np.eye(ndim)
            transform_all[:3, :3] = self.transform

            hess_new = transform_all @ mpt.hess @ transform_all.T

            mpts_new.append(MPoint(center_new, hess_new, mpt.yshift))

        return mpts_new

    def mpts_fromCube(self, mpts):
        mpts_new = []
        for mpt in mpts:
            center_new = mpt.center.copy()
            center_new[:3] = linear_transform(mpt.center[:3].reshape(1, -1), matrix=self.transform, outshift=self.init).reshape(-1,)

            ndim = center_new.shape[0]
            transform_all = np.eye(ndim)
            transform_all[:3, :3] = self.inv_transform

            hess_new = transform_all @ mpt.hess @ transform_all.T

            mpts_new.append(MPoint(center_new, hess_new, mpt.yshift))

        return mpts_new

def multiply_traindata(xall_cube, yall, kx=0, ky=0):
    npt = yall.shape[0]
    nx = 2*kx+1
    ny = 2*ky+1
    yall_ = np.tile(yall, (nx*ny,))
    xall_cube_ = np.tile(xall_cube, (nx*ny,1))
    for ix in range(0, nx):
        for iy in range(0, ny):
            xall_cube_[(ix*ny+iy)*npt:(ix*ny+iy+1)*npt, :2] += np.array([ix-(nx-1)//2, iy-(ny-1)//2])

    return xall_cube_, yall_

class Quadratic():
    def __init__(self, center, hess):
        self.center = center
        self.hess = hess

        return

    def __call__(self, x):
        nsam = x.shape[0]
        yy = np.empty(nsam,)
        for i in range(nsam):
            yy[i] = 0.5 * np.dot(x[i, :] - self.center, np.dot(self.hess, x[i, :] - self.center))

        return yy


class MPoint():
    def __init__(self, center, hess, yshift):
        self.center = center
        self.hess = hess
        self.yshift = yshift

        return

    def __repr__(self):

        rep = f"StatPoint({self.center})"

        return rep


class SModel():
    def __init__(self, ptmodel, mpt, rhomb, expon=True):
        self.ptmodel = ptmodel
        self.mpt = mpt
        self.rhomb = rhomb
        self.expon = expon

        return

    def __call__(self, x):

        x_ = x.copy()
        x_[:, :3] = self.rhomb.toCube(x[:, :3], xyfold=True)
        mpt_ = self.rhomb.mpts_toCube([self.mpt], xyfold=True)[0]


        quad = Quadratic(mpt_.center, mpt_.hess)
        ypred = self.ptmodel(torch.from_numpy(x_ - mpt_.center).double()).detach().numpy().reshape(-1,)
        if self.expon:
            y = mpt_.yshift + quad(x_) * np.exp(ypred)
        else:
            y = mpt_.yshift + quad(x_) * ypred
        return y


class ZModel():
    def __init__(self, mpt):
        self.mpt = mpt

        return

    def __call__(self, x):

        quad = Quadratic(self.mpt.center, self.mpt.hess)
        return self.mpt.yshift + quad(x)


class WFcn():
    def __init__(self, mpts, eps, rhomb=None):
        self.mpts = mpts
        self.centers = np.array([mpt.center for mpt in self.mpts])
        self.eps = eps
        self.rhomb = rhomb

        return

    def __call__(self, x):

        x_ = x.copy()
        centers_ = self.centers.copy()

        if self.rhomb is not None:
            x_[:, :3] = self.rhomb.toCube(x[:, :3], xyfold=True)
            centers_[:, :3] = self.rhomb.toCube(self.centers[:, :3], xyfold=True)

        dists = cdist(x_, centers_)
        scales = np.exp(-dists / self.eps)
        scales /= np.sum(scales, axis=1).reshape(-1, 1)
        return scales


class MultiModelTch(torch.nn.Module):
    def __init__(self, models, wfcn=None, cfcn=None):
        super(MultiModelTch, self).__init__()
        self.models = models
        self.nmod = len(self.models)

        assert(wfcn is not None or cfcn is not None)

        if wfcn is not None:
            assert(cfcn is None)
            self.wflag = True
            self.wfcn = wfcn
        if cfcn is not None:
            assert(wfcn is None)
            self.wflag = False
            self.cfcn = cfcn

    def set_wfcn(self, wfcn):
        self.wfcn = wfcn

        return

    def forward(self, x):

        if self.wflag:
            val = self.wfcn(x)[:, 0] * self.models[0](x).reshape(-1,)
            summ = self.wfcn(x)[:, 0]
            for j in range(1, self.nmod):
                val += self.wfcn(x)[:, j] * self.models[j](x).reshape(-1,)
                summ += self.wfcn(x)[:, j]
            return val / summ

        else:
            y = np.empty((x.shape[0]))
            for j in np.unique(self.cfcn(x)):
                y[self.cfcn(x) == j] = self.models[j](x[self.cfcn(x) == j, :]).reshape(-1,)
            return y



class MultiModel(object):
    def __init__(self, models, wfcn=None, cfcn=None):
        super(MultiModel, self).__init__()
        self.models = models
        self.nmod = len(self.models)

        assert(wfcn is not None or cfcn is not None)

        if wfcn is not None:
            assert(cfcn is None)
            self.wflag = True
            self.wfcn = wfcn
        if cfcn is not None:
            assert(wfcn is None)
            self.wflag = False
            self.cfcn = cfcn

    def __call__(self, x):

        if self.wflag:
            val = self.wfcn(x)[:, 0] * self.models[0](x).reshape(-1,)
            summ = self.wfcn(x)[:, 0]
            for j in range(1, self.nmod):
                val += self.wfcn(x)[:, j] * self.models[j](x).reshape(-1,)
                summ += self.wfcn(x)[:, j]
            return val / summ

        else:
            y = np.empty((x.shape[0]))
            for j in np.unique(self.cfcn(x)):
                y[self.cfcn(x) == j] = self.models[j](x[self.cfcn(x) == j, :]).reshape(-1,)
            return y


def ifcn(x):
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)

    center = np.array([3.5])
    hess = np.array([[1.0]])
    yshift = 0.0
    pt1 = MPoint(center, hess, yshift)
    zm1 = ZModel(pt1)

    center = np.array([7.5])
    hess = np.array([[9.0]])
    yshift = 0.0
    pt2 = MPoint(center, hess, yshift)
    zm2 = ZModel(pt2)

    mpts = [pt1, pt2]
    eps = 0.5
    wfcn = WFcn(mpts, eps)

    mm = MultiModel([zm1, zm2], wfcn=wfcn)

    return mm(x)



##########################################################################################
##########################################################################################
##########################################################################################

class XYMap():
    """docstring for XYMap"""
    def __init__(self):
        return

    def fwd(self, x, y):
        raise NotImplementedError("Base XYMap forward call not implemented")

    def inv(self, x,y):
        raise NotImplementedError("Base XYMap inverse not implemented")

class Identity(XYMap):
    """docstring for Identity"""
    def __init__(self):
        super(Identity, self).__init__()
    def fwd(self, x, y):
        return x, y
    def inv(self, xn, yn):
        return xn, yn

class MPNNMap(XYMap):
    """docstring for MPNNMap"""
    def __init__(self, mapparams):
        super(MPNNMap, self).__init__()
        self.center, self.hess, self.yshift = mapparams
    def fwd(self, x, y):

        quad = Quadratic(self.center, self.hess)

        ynew = (y - self.yshift) / quad(x)
        #print("AAAA ", np.min(ynew), np.min(quad(x)))
        ynew = np.log(ynew)
        xnew = x - self.center

        return xnew, ynew

    def inv(self, xnew, ynew):
        quad = Quadratic(self.center, self.hess)

        x = xnew + self.center
        y = self.yshift + quad(x) * np.exp(ynew)

        return x, y


##########################################################################################
##########################################################################################
##########################################################################################

class MPNNet(MLPBase):

    def __init__(self, nnmodel, in_dim, center, hessian, yshift):
        super(MPNNet, self).__init__(in_dim, 1)
        self.center = center
        self.hessian = hessian
        self.chol = torch.linalg.cholesky(hessian).transpose(-2, -1).conj()  # upper cholesky
        self.yshift = yshift

        self.nnmodel = nnmodel

    def forward(self, x):
        nx = x.shape[0]
        Ux = torch.matmul(x - self.center, self.chol)
        factor2 = torch.sum(Ux**2, dim=1).view(-1, 1)
        #factor = torch.linalg.vector_norm(Ux, dim=1).view(-1,1)
        #factor2 = factor**2
        #print(x.shape, Ux.shape, factor.shape)

        # return 0.5 * torch.exp(torch.sum(x-self.center,1).view(-1,1)*self.nnmodel(x-self.center)) * factor2 +self.yshift
        return 0.5 * torch.exp(self.nnmodel(x - self.center)) * factor2 + self.yshift



def plot_integrand_surr(xtrn, ytrn, ytrn_pred, xtst, ytst, ytst_pred, figname=None, showtest=True):
    Trange = np.arange(400, 1401, 200)
    plt.figure(figsize=(16, 10))

    ir = 2
    ic = 3
    cic = 1
    for T in Trange:
        beta = 1. / (kB * T)

        eytrn = np.exp(-beta * ytrn)
        eytrn_pred = np.exp(-beta * ytrn_pred)

        eytst = np.exp(-beta * ytst)
        eytst_pred = np.exp(-beta * ytst_pred)

        err = rel_l2(eytst_pred, eytst)
        ntr = xtrn.shape[0]
        print(ntr, T, err)
        nts = ytst.shape[0]

        plt.subplot(ir, ic, cic)
        plt.plot(eytrn, eytrn_pred, 'go', markeredgecolor='black',
                 label='Train N$_{trn}$ = ' + str(ntr))
        if showtest:
            plt.plot(eytst, eytst_pred, 'ro', markeredgecolor='black',
                     label='Test  N$_{tst}$ = ' + str(nts))
        plt.gca().set_xlabel(r'$e^{-E/kT}$', fontsize=20)
        plt.gca().set_ylabel(r'$e^{-E_s/kT}$', fontsize=20)
        if cic == 1 and showtest:
            plt.gca().legend(fontsize=12)
        plt.gca().set_title('T=' + str(T) + '   Rel. RMSE=' + "{:0.3f}".format(err), fontsize=15)
        ymin = min(np.min(eytst), np.min(eytst_pred), np.min(eytrn), np.min(eytrn_pred))
        ymax = max(np.min(eytst), np.max(eytst_pred), np.max(eytrn), np.max(eytrn_pred))
        plt.plot([ymin, ymax], [ymin, ymax], 'k--', lw=1)
        plt.gca().axis('equal')
        # plt.yscale('log')
        # plt.xscale('log')
        # plt.gca().set_xlim([ymin, ymax])
        # plt.gca().set_ylim([ymin, ymax])

        #err = np.sqrt(np.mean((eytest_pred-eytest)**2) / np.mean(eytest**2))
        cic += 1

    if not showtest:
        plt.gcf().suptitle('N$_{trn}$ = ' + str(ntr), x=0.05, y=1.0, color='g', fontsize=15)

    plt.gcf().tight_layout(pad=1.5)
    if figname is None:
        plt.savefig('fit_integrands_N' + str(ntr) + '.png')
    else:
        plt.savefig(figname)
    plt.clf()

def plot_xdata(xall, mpts=None, pnames=None, every=1):
    ndim = xall.shape[1]

    if pnames is None:
        pnames = [f'p{j}' for j in range(1, ndim+1)]

    for idim in range(ndim):
        for jdim in range(idim+1, ndim):
            plt.figure(figsize=(10,10))
            plt.plot(xall[::every, idim], xall[::every, jdim], 'o', alpha=0.5, markeredgecolor='w')

            if mpts is not None:
                for mpt in mpts:
                    plt.plot(mpt.center[idim], mpt.center[jdim], 'o', markersize=16, zorder=10000)

            plt.xlabel(f'{pnames[idim]}')
            plt.ylabel(f'{pnames[jdim]}')
            plt.savefig(f'xtrain_{idim}_{jdim}.png')
            plt.clf()
