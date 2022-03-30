#!/usr/bin/env python

import os
import torch
import string
import random
import itertools
import pickle as pk
import numpy as np
import matplotlib as mpl
import matplotlib.tri as tria
import matplotlib.pyplot as plt


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

def read_textlist(filename, nsize):

    if os.path.exists(filename):
        with open(filename) as f:
            names = f.read().splitlines()
            assert(len(names) == nsize)
    else:
        names = ['# ' + str(i) for i in range(1, nsize + 1)]

    return names


def set_colors(npar):
    """ Sets a list of different colors of requested length, as rgb triples"""
    colors = []
    pp = 1 + int(npar / 6)
    for i in range(npar):
        c = 1 - float(int((i / 6)) / pp)
        b = np.empty((3))
        for jj in range(3):
            b[jj] = c * int(i % 3 == jj)
        a = int(int(i % 6) / 3)
        colors.append(((1 - a) * b[2] + a * (c - b[2]),
                       (1 - a) * b[1] + a * (c - b[1]),
                       (1 - a) * b[0] + a * (c - b[0])))

    return colors


def random_str(n):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(n))

def tch(arr):
    #x = torch.from_numpy(arr).double()
    x = torch.tensor(arr, requires_grad=False)
    return x

def npy(arr):
    return arr.data.numpy()

def savepk(sobj, nameprefix='savestate'):
    pk.dump(sobj, open(nameprefix + '.pk', 'wb'), 1) #-1 is default protocol, use 2 for lower versions of python?


def loadpk(nameprefix='savestate'):
    return pk.load(open(nameprefix + '.pk', 'rb'))

def rel_l2(predictions, targets):
    return np.linalg.norm(predictions - targets) / np.linalg.norm(targets)

def cartes_list(somelists):

    final_list = []
    for element in itertools.product(*somelists):
        final_list.append(element)

    return final_list

def linear_transform(x, inshift=None, matrix=None, outshift=None):
    npt, dim = x.shape
    if inshift is None:
        inshift = np.zeros((dim,))
    assert(inshift.shape[0]==dim)
    if outshift is None:
        outshift = np.zeros((dim,))
    assert(outshift.shape[0]==dim)
    if matrix is None:
        matrix = np.eye(dim)
    assert(matrix.shape[0]==dim)
    assert(matrix.shape[1]==dim)

    xnew = np.dot(x-inshift, matrix) + outshift

    #xnew = 2*xnew - 1
    return xnew





def plot_2d_tri(x, z, nlev=22, ax=None, cbar_lims=None):
    triang = tria.Triangulation(x[:, 0], x[:, 1])

    # # if you want to mask some parts, uncomment below
    # xmid = xy[triang.triangles,0].mean(axis=1)
    # ymid = xy[triang.triangles,1].mean(axis=1)
    # mask = np.where(xmid*xmid + ymid*ymid < 1, 1, 0)
    # triang.set_mask(mask)

    if cbar_lims is None:
        levs = np.linspace(z.min(), z.max(), nlev)
    else:
        levs = np.linspace(cbar_lims[0], cbar_lims[1], nlev)

    if ax is None:
        plt.tricontourf(triang, z, levs, extend="both")
        cc=plt.colorbar()
        plt.gca().grid(False)

        return plt.gca()
    else:
        pp = ax.tricontourf(triang, z, levs, extend="both")
        cc=plt.colorbar(pp, ax=ax)
        ax.grid(False)

        return ax


def plot_yx(x, y, rowcols=None, ylabel='', xlabels=None,
            log=False, filename='eda.png',
            ypad=0.3, gridshow=True, ms=2):

    nsam, ndim = x.shape
    assert(nsam==y.shape[0])

    if rowcols is None:
        rows = 3
        cols = (ndim // 3) + 1
    else:
        rows, cols = rowcols



    fig, axes = plt.subplots(rows, cols, figsize=(8*cols,(3+ypad)*rows),
                             gridspec_kw={'hspace': ypad, 'wspace': 0.3})
    #fig.suptitle('Horizontally stacked subplots')

    axes=axes.reshape(rows, cols)

    axes = axes.T
    #print(axes.shape)
    for i in range(ndim):
        ih = i % cols
        iv = i // cols
        axes[ih, iv].plot(x[:, i], y, 'o', ms=ms)
        axes[ih, iv].set_xlabel(xlabels[i])
        axes[ih, iv].set_ylabel(ylabel)
        #axes[ih, iv].set_ylim(ymin=-0.05, ymax=0.5)
        axes[ih, iv].grid(gridshow)
        if log:
            axes[ih, iv].set_yscale('log')

    for i in range(ndim, cols*rows):
        ih = i % cols
        iv = i // cols
        axes[ih, iv].remove()

    plt.savefig(filename)

    #plt.gcf().clear()
    return


def plot_dm(datas, models, errorbars=[], labels=[],
            axes_labels=['Model', 'Apprx'], figname='dm.eps',
            showplot=False, legendpos='in', msize=4):


    """Plots data-vs-model and overlays y=x"""
    if errorbars == []:
        erb = False
    else:
        erb = True

    custom_xlabel = axes_labels[0]
    custom_ylabel = axes_labels[1]

    if legendpos == 'in':
        fig = plt.figure(figsize=(10, 10))
    elif legendpos == 'out':
        fig = plt.figure(figsize=(14, 10))
        fig.add_axes([0.1, 0.1, 0.6, 0.8])

    ncase = len(datas)
    if labels == []:
        labels = [''] * ncase

    # Create colors list
    colors = set_colors(ncase)
    yy = np.empty((0, 1))
    for i in range(ncase):
        data = datas[i]
        model = models[i]
        if erb:
            erbl, erbh = errorbars[i]
        npts = data.shape[0]
        neach = 1
        if (data.ndim > 1):
            neach = data.shape[1]

        # neb=model.shape[1]-1# errbars not implemented yet

        ddata = data.reshape(npts, neach)

        for j in range(neach):
            yy = np.append(yy, ddata[:, j])
            if (erb):
                plt.errorbar(ddata[:, j], model, yerr=[erbl, erbh],
                             fmt='o', markersize=2, ecolor='grey')
            plt.plot(ddata[:, j], model, 'o', color=colors[i], label=labels[i], markersize=msize)

    delt = 0.1 * (yy.max() - yy.min())
    minmax = [yy.min() - delt, yy.max() + delt]
    plt.plot(minmax, minmax, 'k--', linewidth=1.5, label='y=x')

    plt.xlabel(custom_xlabel)
    plt.ylabel(custom_ylabel)
    # plt.title('Data vs Model')
    if legendpos == 'in':
        plt.legend()
    elif legendpos == 'out':
        plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5),
                   ncol=1, fancybox=True, shadow=True)


    # plt.xscale('log')
    # plt.yscale('log')

    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.axis('scaled')
    #plt.axis('equal')
    #plt.gca().set_aspect('equal', adjustable='box')
    # Trying to make sure both axis have the same number of ticks
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(7))
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(7))
    plt.savefig(figname)
    if showplot:
        plt.show()
    plt.clf()
