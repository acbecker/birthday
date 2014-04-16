import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import emcee
from scipy import linalg
from sklearn.gaussian_process import GaussianProcess
from sklearn.gaussian_process.gaussian_process import l1_cross_distances
from sklearn.utils import array2d, check_random_state, check_arrays
from sklearn.metrics.pairwise import manhattan_distances

YEAR = 365.25
MACHINE_EPSILON = np.finfo(np.double).eps

# Covariance Functions
def squared_scaled_exponential(theta, x):
    ssq, lsq = theta
    return ssq * np.exp(-x**2/lsq)
def periodic_exponential(theta, x):
    ssq, lsq, period = theta
    return ssq * np.exp(-2*np.sin(x*np.pi/period)**2 / lsq)

# Core class derived from sklearn.gaussian_process.GaussianProcess
class BirthdayAnalysis(GaussianProcess):
    def __init__(self, **args):
        GaussianProcess.__init__(self, **args)
        self.corr  = self.covariance
        self.raw_X = None
        self.raw_y = None
        self.getData()

        self.lsq1  = 365**2
        self.lsq2  = 10**2
        self.lsq31 = 2**2
        self.lsq32 = 20**2
        self.p3    = 7.0
        self.lsq41 = 100**2
        self.lsq42 = 1000**2
        self.p4    = 365.25

    def covariance(self, theta, x):
        ssq1, ssq2, ssq3, ssq4 = theta
        cov1 = squared_scaled_exponential((ssq1, self.lsq1), x)
        cov2 = squared_scaled_exponential((ssq2, self.lsq2), x)
        cov3 = periodic_exponential((ssq3, self.lsq31, self.p3), x) * squared_scaled_exponential((1.0, self.lsq32), x)
        cov4 = periodic_exponential((ssq4, self.lsq41, self.p4), x) * squared_scaled_exponential((1.0, self.lsq42), x)
        return cov1+cov2+cov3+cov4

    def predict_by_component(self, X, eval_MSE=False):
        ssq1, ssq2, ssq3, ssq4 = self.theta_        
        cov1 = lambda x: squared_scaled_exponential((ssq1, self.lsq1), x)
        cov2 = lambda x: squared_scaled_exponential((ssq2, self.lsq2), x)
        cov3 = lambda x: periodic_exponential((ssq3, self.lsq31, self.p3), x) * squared_scaled_exponential((1.0, self.lsq32), x)
        cov4 = lambda x: periodic_exponential((ssq4, self.lsq41, self.p4), x) * squared_scaled_exponential((1.0, self.lsq42), x)

        pred1 = self.predict(X, corr=cov1, eval_MSE=eval_MSE)
        pred2 = self.predict(X, corr=cov2, eval_MSE=eval_MSE)
        pred3 = self.predict(X, corr=cov3, eval_MSE=eval_MSE)
        pred4 = self.predict(X, corr=cov4, eval_MSE=eval_MSE)

        return pred1, pred2, pred3, pred4

    def getData(self):
        datfile = os.path.join(os.path.dirname(__file__), "birthdates-1968-1988.csv")
        df = pd.read_csv(datfile, "df", delimiter=",",
                         parse_dates={"date": [0,1,2]}, index_col="date")
        X = np.atleast_2d([(x-df.index[0]).days for x in df.index]).T
        y = df.births.values
        self.raw_X = X
        self.raw_y = y

    def compareToSklearn(self, npts=-1, doPlot=True):
        gp    = GaussianProcess(corr="squared_exponential", regr="constant",
                                verbose=True, theta0=365.0, thetaL=1e-1, thetaU=1000)
        X = self.raw_X[:npts]
        y = self.raw_y[:npts]
        gp = gp.fit(X, y)
        print gp, gp.theta_
        xeval = np.atleast_2d(np.linspace(X.min(), X.max(), 1000)).T
        ypred, mse = gp.predict(xeval, eval_MSE=True)
        sigma = np.sqrt(mse)
        if doPlot:
            fig = plt.figure()
            plt.plot(X, y, "ro")
            plt.plot(xeval, ypred, "b-")
            plt.fill_between(xeval[:,0], ypred-sigma, ypred+sigma, facecolor='blue', alpha=0.25)
            plt.show()
        return xeval, ypred, sigma

    def fit(self, X=None, y=None, npts=-1):
        """Override for our subclass"""
        if X is None and y is None:
            X = self.raw_X[:npts]
            y = self.raw_y[:npts]
        else:
            return
            
        # Run input checks
        self._check_params()
        self.random_state = check_random_state(self.random_state)

        # Force data to 2D numpy.array
        X = array2d(X)
        y = np.asarray(y)
        self.y_ndim_ = y.ndim
        if y.ndim == 1:
            y = y[:, np.newaxis]
        X, y = check_arrays(X, y)

        # Check shapes of DOE & observations
        n_samples, n_features = X.shape
        _, n_targets = y.shape

        # Run input checks
        self._check_params(n_samples)

        # Normalize data 
        # Don't scale x-axis so we understand timescales
        X_mean = np.zeros(1)
        X_std = np.ones(1)
        if self.normalize:
            y_mean = np.mean(y, axis=0)
            y_std = np.std(y, axis=0)
            y_std[y_std == 0.] = 1.
            y = (y - y_mean) / y_std
        else:
            y_mean = np.zeros(1)
            y_std = np.ones(1)

        # Calculate matrix of distances D between samples
        D, ij = l1_cross_distances(X)
        if (np.min(np.sum(D, axis=1)) == 0.):
            raise Exception("Multiple input features cannot have the same"
                            " value.")

        # Set attributes
        self.X = X
        self.y = y
        self.D = D
        self.ij = ij
        self.X_mean, self.X_std = X_mean, X_std
        self.y_mean, self.y_std = y_mean, y_std

        # Determine Gaussian Process model parameters
        self.reduced_likelihood_function_value_, par = \
                                                 self.reduced_likelihood_function()

        # Stuff needed for predict
        self.R = par['R']
        self.C = par['C']

        return self

    def reduced_likelihood_function(self, theta=None):
        """ Override for our base class, use emcee to sample posterior"""
        # Initialize output
        reduced_likelihood_function_value = - np.inf
        par = {}

        # Retrieve data
        n_samples = self.X.shape[0]
        D = self.D
        ij = self.ij
        lnl0 = -0.5 * n_samples * np.log(2 * np.pi)
        
        def lnlike(params, *args):
            ssq1, ssq2, ssq3, ssq4, ssq = params
            D, ij, n_samples = args

            # Priors: Driving term should not be less than zero or greater than 3 sigma
            for ss in (ssq1, ssq2, ssq3, ssq4, ssq):
                if ss < 0 or ss > 9:
                    return -np.inf
            lnp = 0.0

            # Set up R
            r = self.corr(params[:-1], D)
            R = np.eye(n_samples) * (1. + params[-1])
            R[ij[:, 0], ij[:, 1]] = r.ravel()
            R[ij[:, 1], ij[:, 0]] = r.ravel()
        
            # Cholesky decomposition of R
            try:
                C = linalg.cholesky(R, lower=True)
            except:
                return -np.inf

            # The determinant of R is equal to the squared product of the diagonal
            # elements of its Cholesky decomposition C
            detR  = (np.diag(C) ** (2. / n_samples)).prod()

            # Marginal likelihood
            lnl1  = -0.5 * np.log(detR)
            lnl2  = -0.5 * np.dot(np.dot(self.y.T, linalg.inv(R)), self.y)
            print params, lnl0, lnl1, lnl2
            return lnl0 + lnl1 + lnl2 + lnp

        if theta is None:
            guess = (0.7**2, 0.4**2, 0.1**2, 0.1**2, 0.1**2)
        else:
            guess = theta
        
        ndim, nwalkers, nburn, nstep = len(guess), 2*len(guess), 10, 100
        pos = [np.array((guess)) + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]    
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlike, args=(D, ij, n_samples))
        pos, prob, state = sampler.run_mcmc(pos, nburn)
        sampler.reset()
        pos, prob, state = sampler.run_mcmc(pos, nstep, rstate0=state)

        flatProb  = sampler.lnprobability.reshape(np.product(sampler.lnprobability.shape))
        sortIdx   = np.argsort(flatProb)[::-1]
        sortPars  = sampler.flatchain[sortIdx]

        # MAP or median?
        mapPars     = sortPars[0]
        medpars     = np.median(sampler.flatchain, axis=0) # Not guaranteed to be an actual step
        
        # Using MAP solution
        self.theta_ = mapPars[:-1]
        self.nugget = mapPars[-1]
        
        r = self.corr(self.theta_, D)
        R = np.eye(n_samples) * (1. + self.nugget)
        R[ij[:, 0], ij[:, 1]] = r.ravel()
        R[ij[:, 1], ij[:, 0]] = r.ravel()
        C = linalg.cholesky(R, lower=True)
        par['C'] = C
        par['R'] = R
        reduced_likelihood_function_value = flatProb[sortIdx[0]]
        
        return reduced_likelihood_function_value, par

    def predict(self, X, corr=None, eval_MSE=False):
        if corr is None:
            corr = lambda x: self.corr(self.theta_, x)
            
        # Check input shapes
        X = array2d(X)
        n_eval, _ = X.shape
        n_samples, n_features = self.X.shape
        n_samples_y, n_targets = self.y.shape

        # Run input checks
        self._check_params(n_samples)

        # Normalize input
        X = (X - self.X_mean) / self.X_std

        # Covariance of new data with old
        dx = manhattan_distances(X, Y=self.X, sum_over_features=False)
        Rcross = corr(dx).reshape(n_eval, n_samples)

        Rinv = linalg.inv(self.R)
        # Scaled predictor
        y_ = np.dot(Rcross, np.dot(Rinv, self.y))
        # Predictor
        y = (self.y_mean + self.y_std * y_).reshape(n_eval, n_targets)

        if eval_MSE:
            D, ij = l1_cross_distances(X)
            r = corr(D)
            Rpred = np.eye(n_eval) * (1. + self.nugget)
            Rpred[ij[:, 0], ij[:, 1]] = r.ravel()
            Rpred[ij[:, 1], ij[:, 0]] = r.ravel()
            var = Rpred - np.dot(Rcross, np.dot(Rinv, Rcross.T))
            return y, var.diagonal()
        return y

if __name__ == "__main__":
    npts = 365*3

    bda  = BirthdayAnalysis()
    #bda.compareToSklearn(npts=npts)
    bda.fit(npts=npts)
    xeval = np.atleast_2d(np.linspace(bda.X.min(), bda.X.max(), 1000)).T
    ypred, var = bda.predict(xeval, eval_MSE=True)
    sigma = np.sqrt(var)
    plt.plot(bda.raw_X, bda.raw_y, "ro")
    plt.plot(xeval, ypred, "b-")
    plt.fill_between(xeval[:,0], ypred[:,0]-sigma, ypred[:,0]+sigma, facecolor='blue', alpha=0.25)

    ypred1, ypred2, ypred3, ypred4 = bda.predict_by_component(xeval, eval_MSE=False)
    plt.plot(xeval, ypred1, "g-")
    plt.plot(xeval, ypred2, "r-")
    plt.plot(xeval, ypred3, "k-")
    plt.plot(xeval, ypred4, "m-")
    
    import pdb; pdb.set_trace()
    plt.show()
    
