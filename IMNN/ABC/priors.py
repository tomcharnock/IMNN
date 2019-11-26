"""Truncated Gaussian prior for use in IMNN ABC

This module provides the Truncated Gaussian prior and functions to draw and
calculate the probability distribution function value when doing ABC with IMNN.

TODO
____
This is a early update to make the ABC module work with TF2, this should be
properly ported at some point soon.
"""


__version__ = '0.2a1'
__author__ = "Justin Alsing and Tom Charnock"


import numpy as np
from scipy.stats import multivariate_normal


class TruncatedGaussian():
    """Truncated Gaussian distribution to be drawn from during ABC with IMNN

    This module provides the functions to draw and calculate the value of the
    distribution function for use in the ABC when using the IMNN. It is based
    on Tom Charnock's version of the prior disrtibution code in Justin Alsing's
    delfi code.

    Attributes
    __________
    mean : ndarray
        the value of the mean of the distribution
    C : ndarray
        the covariance matrix for the multivariate normal distribution
    lower : ndarray
        the lower cut for the Gaussian for each parameter value
    upper : ndarray
        the upper cut for the Gaussian for each parameter value
    L : ndarray
        the cholesky decomposition of the covariance matrix
    """
    def __init__(self, mean, C, lower, upper):
        """Initialisation routine for the truncated Gaussian

        Parameters
        __________
        mean : ndarray
            the value of the mean of the distribution
        C : ndarray
            the covariance matrix for the multivariate normal distribution
        lower : ndarray
            the lower cut for the Gaussian for each parameter value
        upper : ndarray
            the upper cut for the Gaussian for each parameter value
        """
        self.mean = mean
        self.C = C
        self.lower = lower
        self.upper = upper
        self.L = np.linalg.cholesky(C)

    def uniform(self, x):
        """Uniform distribution

        This simultaneously checks whether all values are within the allowed
        truncation range and returns zero if not and the value of the uniform
        if the parameter values are all allowed.

        Parameters
        __________
        x : ndarray
            the parameter values to check
        up : bool
            True if any values are above the allowed range
        down : bool
            True is any values are below the allowed range

        Returns
        _______
        float
            returns zero if any parameter values are not allowed or the value
            of the uniform prior if they are all allowed
        """
        up = np.prod(x >= self.upper[np.newaxis, :]).astype(np.bool)
        down = np.prod(x <= self.lower[np.newaxis, :]).astype(np.bool)
        if up or down:
            return 0.
        else:
            return np.prod(self.upper - self.lower)

    def draw(self, to_draw=1):
        """A parameter draw from the distribution

        As many draws can be made from the prior simultaneously as needed.

        Parameters
        __________
        to_draw : int, optional
            the number of draws to make at one time
        x_keep : ndarray
            the drawn values which are within the truncation bounds
        squeeze : bool
            whether to squeeze the dimension if only one parameter is drawn
        cov : ndarray
            random variable which is multiplied by cholesky to get a sample of
            a parameter value
        x : ndarray
            the newly drawn samples
        up : list of bool
            each element is True for drawn parameters are below upper threshold
        down : list of bool
            each element is True for drawn parameters are above lower threshold
        ind : list of int
            indices where both up and down are True (allowed parameter value)

        Returns
        _______
        ndarray
            the array of allowed parameters drawn from the truncated Gaussian
        """
        x_keep = None
        if to_draw == 1:
            squeeze = True
        else:
            squeeze = False
        while to_draw > 0:
            cov = np.random.normal(0, 1, (to_draw, self.mean.shape[0]))
            x = self.mean[np.newaxis, :] + np.dot(cov, self.L)
            up = x <= self.upper[np.newaxis, :]
            down = x >= self.lower[np.newaxis, :]
            ind = np.argwhere(np.all(up * down, axis=1))[:, 0]
            if x_keep is None:
                x_keep = x[ind]
            else:
                x_keep = np.concatenate([x_keep, x[ind]])
            to_draw -= ind.shape[0]
        if squeeze:
            x_keep = np.squeeze(x_keep, 0)
        return x_keep

    def pmc_draw(self):
        """A parameter draws for every sample which needs moving in the PMC

        Parameters
        __________
        x_keep : ndarray
            the drawn values which are within the truncation bounds
        kept : int
            counter for the number of parameter samples which are successful
        change_ind : list of int
            indices of the parameters which are successfully in the proposal
        cov : ndarray
            random variable which is multiplied by cholesky to get a sample of
            a parameter value
        x : ndarray
            the newly drawn samples
        up : list of bool
            each element is True for drawn parameters are below upper threshold
        down : list of bool
            each element is True for drawn parameters are above lower threshold
        keep_ind : list of int
            indices where both up and down are True (allowed parameter value)

        Returns
        _______
        ndarray
            the array of allowed parameters drawn from the truncated Gaussian
        """
        x_keep = np.zeros_like(self.mean)
        kept = 0
        change_ind = np.arange(self.mean.shape[0]).astype(np.int)
        while kept < self.mean.shape[0]:
            cov = np.random.normal(
                0,
                1,
                (change_ind.shape[0], self.mean.shape[1]))
            x = self.mean[change_ind] + np.dot(cov, self.L)
            up = x <= self.upper[np.newaxis, :]
            down = x >= self.lower[np.newaxis, :]
            keep_ind = np.argwhere(np.all(up * down, axis=1))
            kept += keep_ind.shape[0]
            if keep_ind.shape[0] > 0:
                x_keep[keep_ind] = x[keep_ind]
                change_ind = change_ind[np.isin(change_ind,
                                                keep_ind,
                                                invert=True)]
        return x_keep

    def pdf(self, x):
        """The value of the probablity distribution function for the prior

        Parameters
        __________
        x : ndarray
            value at which to calculate the value of the probability
            distribution at

        Returns
        _______
        ndarray
            the value of the probability distribution
        """
        return self.uniform(x) \
            * multivariate_normal.pdf(
                x,
                mean=self.mean,
                cov=self.C)
