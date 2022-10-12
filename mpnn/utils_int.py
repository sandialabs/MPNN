#!/usr/bin/env python

import numpy as np

from scipy.stats import multivariate_normal, kde
from scipy.integrate import nquad

from .utils_mrv import MCMCRV, GMM



class LinearScaler():
    def __init__(self, shift=None, scale=None):
        self.shift = shift
        self.scale = scale

        return

    def __repr__(self):
        return f"Scaler({self.shift=}, {self.scale=}"

    def __call__(self, x):
        if self.shift is None:
            xs = x - 0.0
        else:
            xs = x - self.shift

        if self.scale is None:
            xs /= 1.0
        else:
            xs /= self.scale

        return xs

    def inv(self, xs):
        if self.scale is None:
            x = xs * 1.0
        else:
            x = xs * self.scale#.reshape(1,-1)

        if self.shift is None:
            x += 0.0
        else:
            x += self.shift

        return x

class Domainizer(LinearScaler):
    def __init__(self, dom):
        super(Domainizer, self).__init__(shift=dom[:,0], scale=dom[:,1]-dom[:,0])
        return


class Integrator(object):
    """docstring for Integrator"""

    def __init__(self):
        super(Integrator, self).__init__()
        return

    def integrate(self, function, domain=None, func_args=None, xdata=None, **kw_args):
        raise NotImplementedError

    def _save(self, a, b, kk):
        if kk in a.keys():
            b[kk] = a[kk].copy()
        else:
            pass


    def integrate_multiple(self, f_arg_list, **kwargs):
        integrals = []
        for i, f_args in enumerate(f_arg_list):
            function, args = f_args
            if i>0:
                self._save(results, args, 'saved')
                self._save(results, kwargs, 'xdata')

            integral, results = self.integrate(function, func_args=args, **kwargs)

            integrals.append(integral)

        return integrals


class IntegratorScipy(Integrator):
    """docstring for IntegratorScipy"""

    def __init__(self):
        super(IntegratorScipy, self).__init__()
        return

    def integrate(self, function, domain=None, func_args=None, epsrel=1.e-5, xdata=None):
        def wrapper(*args):
            func, dim, func_pars = args[-3:]
            func_input = np.array(args[:dim]).reshape(1, dim)
            return func(func_input, **func_pars)[0]
        assert(xdata is None)
        assert(domain is not None)
        dim, two = domain.shape
        integral, err, results = nquad(wrapper,
                                       domain,
                                       args=(function, dim, func_args),
                                       opts={'epsrel': epsrel},
                                       full_output=True)
        results['err'] = err

        return integral, results


class IntegratorMCMC(Integrator):
    """docstring for IntegratorMCMC"""

    def __init__(self):
        super(IntegratorMCMC, self).__init__()
        return

    def integrate(self, function, nmc=100, domain=None, func_args=None, xdata=None):
        def logfunction(x, **args):
            assert(len(x.shape) == 1)
            return np.log(function(x.reshape(1, -1), **args)[0])
        assert(domain is not None)
        dim, two = domain.shape

        if xdata is None:
            rv = MCMCRV(dim, logfunction, param_ini=np.ones((dim,)), nmcmc=10 * nmc)
            xdata = rv.sample(nmc, **func_args)
        ydata = function(xdata, **func_args)

        kde_py = kde.gaussian_kde(xdata.T)
        kde_weight = kde_py(xdata.T)

        integral = np.mean(ydata / kde_weight)

        results = {'err': None, 'neval': nmc, 'xdata': xdata, 'ydata': ydata, 'icw': kde_weight}
        return integral, results


class IntegratorMC(Integrator):
    """docstring for IntegratorMC"""

    def __init__(self, seed=None):
        super(IntegratorMC, self).__init__()
        np.random.seed(seed=seed)
        return

    def integrate(self, function, domain=None, nmc=100, func_args=None, xdata=None):
        assert(domain is not None)
        dim, two = domain.shape

        if xdata is None:
            sc = Domainizer(domain)
            xdata = sc.inv(np.random.rand(nmc, dim))
        ydata, saved = function(xdata, **func_args)

        volume = np.prod(domain[:, 1] - domain[:, 0])
        integral = volume * np.mean(ydata)
        results = {'err': None, 'neval': nmc, 'xdata': xdata, 'ydata': ydata, 'saved': saved}
        return integral, results


class IntegratorWMC(Integrator):
    """docstring for IntegratorWMC"""

    def __init__(self, seed=None):
        super(IntegratorWMC, self).__init__()
        np.random.seed(seed=seed)
        return

    def integrate(self, function, mean=None, cov=None,
                  nmc=100, func_args=None, xdata=None):
        assert(mean is not None)
        if cov is None:
            cov = np.eye(mean.shape[0])

        if xdata is None:
            xdata = np.random.multivariate_normal(mean,
                                              cov,
                                              size=(nmc, ))
        ydata, saved = function(xdata, **func_args)

        gaussian_weight = multivariate_normal.pdf(xdata,
                                                  mean=mean,
                                                  cov=cov)

        integral = np.mean(ydata / gaussian_weight)

        results = {'err': None, 'neval': nmc, 'xdata': xdata, 'ydata': ydata, 'icw': gaussian_weight, 'saved': saved}
        return integral, results


class IntegratorGMM(Integrator):
    """docstring for IntegratorGMM"""

    def __init__(self):
        super(IntegratorGMM, self).__init__()
        return

    def integrate(self, function,
                  weights=None, means=None, covs=None,
                  nmc=100, func_args=None, xdata=None):
        assert(means is not None)
        if weights is None:
            weights = [function(mean.reshape(1,-1), **func_args)[0][0] * np.sqrt(np.linalg.det(cov)) for mean, cov in zip(means, covs)]

        mygmm = GMM(means, covs=covs, weights=weights)

        if xdata is None:
            xdata = mygmm.sample(nmc)
        ydata, saved = function(xdata, **func_args)

        gmm_pdf = mygmm.pdf(xdata)

        integral = np.mean(ydata / gmm_pdf)

        results = {'err': None, 'neval': nmc, 'xdata': xdata, 'ydata': ydata, 'icw': gmm_pdf, 'saved': saved}
        return integral, results


class IntegratorGMMT(Integrator):
    """docstring for IntegratorGMMT"""

    def __init__(self):
        super(IntegratorGMMT, self).__init__()

        return

    def integrate(self, function, domain=None,
                  weights=None, means=None, covs=None,
                  nmc=100, func_args=None, xdata=None):
        assert(means is not None)
        if weights is None:
            weights = [function(mean.reshape(1,-1), **func_args)[0][0] * np.sqrt(np.linalg.det(cov)) for mean, cov in zip(means, covs)]



        mygmm = GMM(means, covs=covs, weights=weights)

        if xdata is None:
            xdata = mygmm.sample_indomain(nmc, domain)
        ydata, saved = function(xdata, **func_args)

        gmm_pdf = mygmm.pdf(xdata)
        volume = mygmm.volume_indomain(domain)
        integral = volume * np.mean(ydata / gmm_pdf)

        results = {'err': None, 'neval': nmc, 'xdata': xdata, 'ydata': ydata, 'icw': gmm_pdf/volume, 'saved': saved}
        return integral, results


