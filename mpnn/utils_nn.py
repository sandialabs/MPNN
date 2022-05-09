#!/usr/bin/env python

import os
import copy
import torch
import string
import random
import numpy as np
import matplotlib.pyplot as plt

from .utils_gen import tch, npy

torch.set_default_dtype(torch.double)


class MLPBase(torch.nn.Module):

    def __init__(self, indim, outdim):
        super(MLPBase, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.model = None
        return

    def forward(self, x):
        if self.model is not None:
            return self.model(x)
        else:
            print("Forward for base is not implemented")
            return
        print("Forward for base is not implemented")
        return

    def predict(self, x, best=True):
        if best:
            return npy(self.best_instance(tch(x)))
        else:
            return npy(self.forward(tch(x)))

    def numpar(self):
        pdim = sum(p.numel() for p in self.parameters())
        return pdim

    def fit(self, xtrn, ytrn, val=None,
            loss_fn='mse', optimizer='adam', wd=0.0,
            lrate=0.1, lmbd=None, batch_size=None, num_batches=None,
            nepochs=5000,
            gradcheck=False, freq_out=100, freq_plot=1000, loss=None, opt=None):

        assert(batch_size is None or num_batches is None)

        # Loss function
        if loss is None:
            if loss_fn == 'mse':
                loss = torch.nn.MSELoss(reduction='mean')
            else:
                print(f"Loss function {loss_fn} is unknown. Exiting.")
                sys.exit()

        # Optimizer selection
        if opt is None:
            if optimizer == 'adam':
                opt = torch.optim.Adam(self.parameters(), lr=lrate, weight_decay=wd)
            elif optimizer == 'sgd':
                opt = torch.optim.SGD(self.parameters(), lr=lrate, weight_decay=wd)
            else:
                print(f"Optimizer {optimizer} is unknown. Exiting.")
                sys.exit()

        # Learning rate schedule
        if lmbd is None:
            def lmbd(epoch): return 1.0
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lmbd)

        ntrn = xtrn.shape[0]
        assert(xtrn.shape[1] == self.indim)

        if batch_size is None:
            if num_batches is None:
                batch_size = ntrn
            else:
                batch_size = ntrn // (num_batches - 1)

        xtrn_ = tch(xtrn)
        ytrn_ = tch(ytrn)

        # Validation data
        if val is None:
            xval, yval = xtrn.copy(), ytrn.copy()
        else:
            xval, yval = val

        xval_ = tch(xval)
        yval_ = tch(yval)

        # Training process
        self.best_fepoch = 0
        self.best_epoch = 0
        self.best_loss = 1.e+100
        self.best_instance = self
        self.ntrn = ntrn # useful to store

        self.history = []
        fepochs = 0
        for t in range(nepochs):
            permutation = torch.randperm(ntrn)
            # for parameter in model.parameters():
            #     print(parameter)
            nsubepochs = len(range(0, ntrn, batch_size))
            for i in range(0, ntrn, batch_size):
                indices = permutation[i:i + batch_size]
                ytrn_pred = self.forward(xtrn_[indices, :])
                with torch.no_grad():
                    yval_pred = self.forward(xval_)

                loss_trn = loss(ytrn_pred, ytrn_[indices, :])
                #loss_val = loss_trn
                with torch.no_grad():
                    loss_val = loss(yval_pred, yval_)
                #loss_trn_full = loss_trn
                if i == 0:  # otherwise too expensive
                    with torch.no_grad():
                        ytrn_pred_full = self.forward(xtrn_)
                        loss_trn_full = loss(ytrn_pred_full, ytrn_)

                fepochs += 1. / nsubepochs

                curr_state = [fepochs + 0.0, loss_trn.item(), loss_trn_full.item(), loss_val.item()]
                crit = loss_val.item()

                if crit < self.best_loss:
                    self.best_loss = crit
                    # Is this dangerous?
                    delattr(self, 'best_instance')
                    self.best_instance = copy.copy(self)

                    self.best_fepoch = fepochs
                    self.best_step = len(self.history)
                    self.best_epoch = t

                self.history.append(curr_state)

                if gradcheck:
                    gc = torch.autograd.gradcheck(self.forward, (xtrn_,),
                                                  eps=1e-2, atol=1e-2)

                opt.zero_grad()
                loss_trn.backward()

                opt.step()
                # print(fepochs)

            scheduler.step()

            if t == 0:
                print('{:>10} {:>10} {:>10} {:>10} {:>10} {:>12} {:>10}'.\
                      format("NEpochs", "NUpdates",
                             "BatchLoss", "TrnLoss", "ValLoss",
                             "BestLoss(Epoch)", "LrnRate"))

            if (t + 1) % freq_out == 0 or t == 0 or t == nepochs - 1:
                tlr = opt.param_groups[0]['lr']
                print(f'{t+1:>10} {len(self.history):>10} {self.history[-1][1]:>10.6f} {self.history[-1][2]:>10.6f} {self.history[-1][3]:>10.6f} {self.best_loss:>12.6f}({self.best_epoch}) {tlr:>10.6f}')
            if t % freq_plot == 0 or t == nepochs - 1:
                self.plot_history()

            if self.best_loss < 1.e-10:
                break

        return self.best_instance

    def printParams(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data)

        return

    def printParamNames(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data.shape)

        return

    def plot_history(self):

        tst_avail = False
        if len(self.history[0]) > 3:
            tst_avail = True

        fepochs = [state[0] for state in self.history]
        losses_trn = [state[1] for state in self.history]
        losses_trn_full = [state[2] for state in self.history]
        if tst_avail:
            losses_tst = [state[3] for state in self.history]

        _ = plt.figure(figsize=(12, 8))

        plt.plot(fepochs, losses_trn, label='Batch loss')
        plt.plot(fepochs, losses_trn_full, label='Training loss')
        plt.plot(self.best_fepoch, self.best_loss, 'ro', markersize=11)
        plt.vlines(self.best_fepoch, 0.0, 2.0, colors=None, linestyles='--')
        if tst_avail:
            plt.plot(fepochs, losses_tst, label='Validation loss')

        plt.legend()
        plt.savefig('loss_history.png')
        plt.yscale('log')
        plt.savefig('loss_history_log.png')
        plt.clf()
        return

    def predict_plot(self, xx_list, yy_list, labels=None, colors=None, iouts=None):
        nlist = len(xx_list)
        assert(nlist==len(yy_list))


        yy_pred_list = []
        for xx in xx_list:
            yy_pred = self.predict(xx)
            yy_pred_list.append(yy_pred)

        nout = yy_pred.shape[1]
        if iouts is None:
            iouts = range(nout)

        if labels is None:
            labels = [f'Set {i+1}' for i in range(nlist)]
        assert(len(labels)==nlist)

        if colors is None:
            colors = ['b', 'g', 'r', 'c', 'm', 'y']*nlist
            colors = colors[:nlist]
        assert(len(colors)==nlist)

        for iout in iouts:
            x1 = [yy[:, iout] for yy in yy_list]
            x2 = [yy[:, iout] for yy in yy_pred_list]

            plot_dm(x1, x2, labels=labels, colors=colors,
                    axes_labels=[f'Model output # {iout+1}', f'Fit output # {iout+1}'],
                    figname='fitdiag_o'+str(iout)+'.png',
                    showplot=False, legendpos='in', msize=13)

        return

    def plot_1d_fits(self, xx_list, yy_list, domain=None, ngr=111, true_model=None, labels=None, colors=None):

        nlist = len(xx_list)
        assert(nlist==len(yy_list))

        if labels is None:
            labels = [f'Set {i+1}' for i in range(nlist)]
        assert(len(labels)==nlist)

        if colors is None:
            colors = ['b', 'g', 'r', 'c', 'm', 'y']*nlist
            colors = colors[:nlist]
        assert(len(colors)==nlist)

        if domain is None:
            xall = functools.reduce(lambda x,y: np.vstack((x,y)), xx_list)
            domain = get_domain(xall)

        mlabel = 'Mean Pred.'

        ndim = xx_list[0].shape[1]
        nout = yy_list[0].shape[1]
        for idim in range(ndim):
            xgrid_ = 0.5 * np.ones((ngr, ndim))
            xgrid_[:, idim] = np.linspace(0., 1., ngr)

            xgrid = scale01ToDom(xgrid_, domain)
            ygrid_pred = self.predict(xgrid)

            for iout in range(nout):

                for j in range(nlist):
                    xx = xx_list[j]
                    yy = yy_list[j]

                    plt.plot(xx[:, idim], yy[:, iout], colors[j]+'o', markersize=13, markeredgecolor='w', label=labels[j])

                if true_model is not None:
                    true = true_model(xgrid, 0.0)
                    plt.plot(xgrid[:, idim], true[:, iout], 'k-', label='Truth', alpha=0.5)


                p, = plt.plot(xgrid[:, idim], ygrid_pred[:, iout], 'm-', linewidth=5, label=mlabel)


                plt.legend()
                plt.xlabel(f'Input # {idim+1}')
                plt.ylabel(f'Output # {iout+1}')
                plt.savefig('fit_d' + str(idim) + '_o' + str(iout) + '.png')
                plt.clf()

        return

class MLP(MLPBase):

    def __init__(self, indim, outdim, hls, biasorno=True,
                 activ='relu', bnorm=False, bnlearn=True, dropout=0.0,
                 final_transform=None):
        super(MLP, self).__init__(indim, outdim)

        self.nlayers = len(hls)
        assert(self.nlayers > 0)
        self.hls = hls
        self.biasorno = biasorno
        self.dropout = dropout
        self.bnorm = bnorm
        self.bnlearn = bnlearn
        self.final_transform = final_transform

        if activ == 'tanh':
            activ_fcn = torch.nn.Tanh()
        elif activ == 'relu':
            activ_fcn = torch.nn.ReLU()
        elif activ == 'sigm':
            activ_fcn = torch.nn.Sigmoid()
        else:
            activ_fcn = torch.nn.Identity()

        modules = []
        modules.append(torch.nn.Linear(self.indim, self.hls[0], self.biasorno))
        if self.dropout > 0.0:
            modules.append(torch.nn.Dropout(p=self.dropout))

        if self.bnorm:
            modules.append(torch.nn.BatchNorm1d(self.hls[0], affine=self.bnlearn))
        for i in range(1, self.nlayers):
            modules.append(activ_fcn)
            modules.append(torch.nn.Linear(self.hls[i - 1], self.hls[i], self.biasorno))
            if self.dropout > 0.0:
                modules.append(torch.nn.Dropout(p=self.dropout))
            if self.bnorm:
                modules.append(torch.nn.BatchNorm1d(self.hls[i], affine=self.bnlearn))


        modules.append(activ_fcn)
        modules.append(torch.nn.Linear(self.hls[-1], self.outdim, bias=self.biasorno))
        if self.dropout > 0.0:
            modules.append(torch.nn.Dropout(p=self.dropout))
        if self.bnorm:
            modules.append(torch.nn.BatchNorm1d(self.outdim, affine=self.bnlearn))

        if self.final_transform=='exp':
            modules.append(Expon())


        self.model = torch.nn.Sequential(*modules)


class Expon(torch.nn.Module):
    def __init__(self):
        super().__init__() # init the base class

    def forward(self, input):
        return torch.exp(input)



class NNWrap():
    def __init__(self, nnmodule):
        self.nnmodel = nnmodule
        pp = self.p_flatten()

    def __call__(self, x):
        xt = tch(x)
        return npy(self.nnmodel.forward(xt))

    def p_flatten(self):
        """
        flattens all parameters into a single column vector. Returns the dictionary to recover them
        :param: parameters: a generator or list of all the parameters
        :return: a dictionary: {"params": [#params, 1],
        "indices": [(start index, end index) for each param] **Note end index in uninclusive**

        """
        l = [torch.flatten(p) for p in self.nnmodel.parameters()]
        self.indices = []
        s = 0
        for p in l:
            size = p.shape[0]
            self.indices.append((s, s+size))
            s += size
        flat_parameter = torch.cat(l).view(-1, 1)

        return flat_parameter

    def p_unflatten(self, flat_parameter, setp=True):
        """
        Gives a list of recovered parameters from their flattened form
        :param flat_params: [#params, 1]
        :param indices: a list detaling the start and end index of each param [(start, end) for param]
        :param model: the model that gives the params with correct shapes
        :return: the params, reshaped to the ones in the model, with the same order as those in the model
        """

        ll = [tch(flat_parameter[s:e]) for (s, e) in self.indices]
        for i, p in enumerate(self.nnmodel.parameters()):
            if len(p.shape)>0:
                ll[i] = ll[i].view(*p.shape)
            if setp:
                p.data = ll[i]

        return ll


###########################################################################
###########################################################################

class PeriodicLoss(torch.nn.Module):

    def __init__(self, loss_params):
        super(PeriodicLoss, self).__init__()
        self.model, self.lam, self.bdry1, self.bdry2 = loss_params

    def forward(self, inputs, targets):
        tmp = (inputs-targets)**2
        fit = torch.mean(tmp)
        penalty = 0.0
        if self.lam>0:
            penalty = self.lam * torch.mean((self.model(self.bdry1)-self.model(self.bdry2))**2)
        loss =  fit + penalty
        #print(fit, penalty)

        return loss
