#!/usr/bin/env python


import sys
import functools
import numpy as np
from itertools import product
from scipy.stats import multivariate_normal


class MRV():
    def __init__(self, pdim):
        self.pdim = pdim
        return

    def __repr__(self):
        return f"Multivariate Random Variable(dim={self.pdim})"

    def sample(self, nsam):
        print("Sampling is not implemented.")
        return

    def pdf(self, x):
        print("PDF function is not implemented.")
        return

    def cdf(self, x):
        print("CDF function is not implemented.")
        return

    def sample_indomain(self, nsam, domain):
        sam = np.empty((0, self.pdim))
        while sam.shape[0]<nsam:
            # TODO need proper checks or maxz to avoid infinite while loop
            sam_check = self.sample(nsam)
            ind = (sam_check>domain[:,0]).all(axis=1)*(sam_check<domain[:,1]).all(axis=1)
            sam = np.vstack((sam, sam_check[ind]))


        return sam[:nsam]

    def volume_indomain(self, domain):
        # see page 3 of https://www.cesarerobotti.com/wp-content/uploads/2019/04/JCGS-KR.pdf
        ii = np.array([i for i in product(range(2), repeat=self.pdim)])
        volume = 0.0
        for i in ii:
            corner = domain[:, 0].copy()
            corner[i==1] = domain[i==1, 1]
            volume += (-1)**(self.pdim-np.sum(i)) * self.cdf(corner)

        return volume



class GMM(MRV):
    """docstring for GMM"""
    def __init__(self, means, covs=None, weights=None):
        super(GMM, self).__init__(len(means[0]))

        self.means = means
        ncl = len(self.means)

        if covs is None:
            self.covs = [np.eye(mean.shape[0]) for mean in self.means]
        else:
            self.covs=covs
        if weights is None:
            self.weights = np.ones((ncl,))/ncl
        else:
            self.weights = weights / np.sum(weights)

        self.size_checks()


        return

    def size_checks(self):
        assert(len(self.weights)==len(self.means))
        assert(len(self.weights)==len(self.covs))
        for mean, cov in zip(self.means, self.covs):
            assert(len(mean)==len(self.means[0]))
            assert(len(mean)==cov.shape[0])
            assert(len(mean)==cov.shape[1])

        return

    def sample(self, nsam):
        nmcs = np.random.multinomial(nsam, self.weights, size=1)[0]
        assert(np.sum(nmcs) == nsam)
        sam = np.empty((0, self.pdim))
        for mean, cov, nmc_this in zip(self.means, self.covs, nmcs):
            sam_this = np.random.multivariate_normal(mean,
                                                       cov,
                                                       size=(int(nmc_this), ))
            sam = np.vstack((sam, sam_this))

        return sam

    def pdf(self, xdata):
        gmm_pdf = 0.0
        for mean, cov, weight in zip(self.means, self.covs, self.weights):
            gmm_pdf += weight * multivariate_normal.pdf(xdata, mean=mean, cov=cov)

        return gmm_pdf

    def cdf(self, xdata):

        gmm_cdf = 0.0
        for mean, cov, weight in zip(self.means, self.covs, self.weights):
            dist = multivariate_normal(mean=mean, cov=cov)
            gmm_cdf += weight * dist.cdf(np.array(xdata))

        return gmm_cdf


class MCMCRV(MRV):
    def __init__(self, pdim, logpost, param_ini=None, nmcmc=10000):
        super(MCMCRV, self).__init__(pdim)
        self.logpost = logpost
        self.param_ini = param_ini
        self.nmcmc = nmcmc
        return

    def sample(self, nsam, **post_info):
        assert(nsam<self.nmcmc//2)
        cov_ini = np.diag(0.1+0.1*np.abs(self.param_ini))

        calib_params={'param_ini': self.param_ini, 'cov_ini': cov_ini,
                      't0': 100, 'tadapt' : 100,
                      'gamma' : 0.1, 'nmcmc' : self.nmcmc}

        calib = AMCMC()
        calib.setParams(**calib_params)

        calib_results = calib.run(self.logpost, **post_info)
        #calib_results = {'chain' : samples, 'mapparams' : cmode,
        #'maxpost' : pmode, 'accrate' : acc_rate, 'logpost' : logposts, 'alphas' : alphas}

        every = (self.nmcmc//2)//nsam
        sam = calib_results['chain'][self.nmcmc//2::every]
        assert(sam.shape[0]==nsam)
        return sam




class AMCMC():
    def __init__(self):
        return

    def setParams(self, param_ini=None, cov_ini=None,
                  t0=100, tadapt=1000,
                  gamma=0.1, nmcmc=10000):
        self.param_ini = param_ini
        self.cov_ini = cov_ini
        self.t0 = t0
        self.tadapt = tadapt
        self.gamma = gamma
        self.nmcmc = nmcmc

    # def setData_calib(self, xd, yd):
    #     self.xd = xd # list of conditions
    #     self.yd = yd # list of arrays per condition

    # Adaptive Markov chain Monte Carlo
    def run(self, logpostFcn, **postInfo):
        cdim = self.param_ini.shape[0]            # chain dimensionality
        cov = np.zeros((cdim, cdim))   # covariance matrix
        samples = np.zeros((self.nmcmc, cdim))  # MCMC samples
        alphas = np.zeros((self.nmcmc,))  # Store alphas (posterior ratios)
        logposts = np.zeros((self.nmcmc,))  # Log-posterior values
        na = 0                        # counter for accepted steps
        sigcv = self.gamma * 2.4**2 / cdim
        samples[0] = self.param_ini                  # first step
        p1 = -logpostFcn(samples[0], **postInfo)  # NEGATIVE logposterior
        pmode = p1  # record MCMC 'mode', which is the current MAP value (maximum posterior)
        cmode = samples[0]  # MAP sample
        acc_rate = 0.0  # Initial acceptance rate

        # Loop over MCMC steps
        for k in range(self.nmcmc - 1):

            # Compute covariance matrix
            if k == 0:
                Xm = samples[0]
            else:
                Xm = (k * Xm + samples[k]) / (k + 1.0)
                rt = (k - 1.0) / k
                st = (k + 1.0) / k**2
                cov = rt * cov + st * np.dot(np.reshape(samples[k] - Xm, (cdim, 1)), np.reshape(samples[k] - Xm, (1, cdim)))
            if k == 0:
                propcov = self.cov_ini
            else:
                if (k > self.t0) and (k % self.tadapt == 0):
                    propcov = sigcv * (cov + 10**(-8) * np.identity(cdim))

            # Generate proposal candidate
            u = np.random.multivariate_normal(samples[k], propcov)
            p2 = -logpostFcn(u, **postInfo)
            #print(u, p1, p2)
            pr = np.exp(p1 - p2)
            alphas[k + 1] = pr
            logposts[k + 1] = -p2

            # Accept...
            if np.random.random_sample() <= pr:
                samples[k + 1] = u
                na = na + 1  # Acceptance counter
                p1 = p2
                if p1 <= pmode:
                    pmode = p1
                    cmode = samples[k + 1]
            # ... or reject
            else:
                samples[k + 1] = samples[k]


            acc_rate = float(na) / (k+1)

            if((k + 2) % (self.nmcmc / 10) == 0) or k == self.nmcmc - 2:
                print('%d / %d completed, acceptance rate %lg' % (k + 2, self.nmcmc, acc_rate))

        mcmc_results = {'chain' : samples, 'mapparams' : cmode,
        'maxpost' : pmode, 'accrate' : acc_rate, 'logpost' : logposts, 'alphas' : alphas}

        return mcmc_results
