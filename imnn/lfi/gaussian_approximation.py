import jax
import jax.numpy as np
from jax.scipy.stats import norm, multivariate_normal
from imnn.lfi import LikelihoodFreeInference


class GaussianApproximation(LikelihoodFreeInference):
    """Uses Fisher information and parameter estimates approximate marginals

    Since the inverse of the Fisher information matrix describes the minimum
    variance of some estimator we can use it to make an approximate (Gaussian-
    distributed) estimate of the target distribution. Note that this will not
    reflect the true shape of the target distribution and as likely to
    underestimate the distribution as overestimate it. Furthermore, if the
    Fisher information matrix is calculated far from the estimate of the
    parameter values then its value may not be representative of the Fisher
    information at that position and so the variance estimated from its inverse
    be incorrect.

    Parameters
    ----------
    parameter_estimates: float(n_targets, n_params)
        The parameter estimates of each target data
    invF: float(n_targets, n_params, n_params)
        The inverse Fisher information matrix for each target
    marginals: list of lists
        The 1D and 2D marginal distribution for each target

    Todo
    ----
    type checking and pytests need implementing
    """
    def __init__(self, parameter_estimates, invF, prior, gridsize=100):
        """Constructor method

        Parameters
        ----------
        parameter_estimates: float(n_targets, n_params)
            The parameter estimates of each target data
        invF: float(n_targets, n_params, n_params)
            The inverse Fisher information matrix for each target
        prior: fn
            A prior distribution which can be evaluated and sampled from
            (should also contain a ``low`` and a ``high`` attribute with
            appropriate ranges)
        gridsize : int or list, default=100
            The number of grid points to evaluate the marginal distribution on
            for every parameter (int) or each parameter (list)
        """
        super().__init__(
            prior=prior,
            gridsize=gridsize)
        if len(parameter_estimates.shape) == 0:
            parameter_estimates = np.expand_dims(parameter_estimates, 0)
        if len(parameter_estimates.shape) == 1:
            parameter_estimates = np.expand_dims(parameter_estimates, 0)
        self.parameter_estimates = parameter_estimates
        self.n_targets = self.parameter_estimates.shape[0]
        self.n_params = self.parameter_estimates.shape[-1]
        self.invF = invF
        self.marginals = self.get_marginals()

    def get_marginals(self, parameter_estimates=None, invF=None, ranges=None,
                      gridsize=None):
        """
        Creates list of 1D and 2D marginal distributions ready for plotting

        The marginal distribution lists from full distribution array. For every
        parameter the full distribution is summed over every other parameter to
        get the 1D marginals and for every combination the 2D marginals are
        calculated by summing over the remaining parameters. The list is made
        up of a list of n_params lists which contain n_columns number of
        objects. The value of the distribution comes from

        Parameters
        ----------
        parameter_estimates: float(n_targets, n_params) or None, default=None
            The parameter estimates of each target data. If None the class
            instance parameter estimates are used
        invF: float(n_targets, n_params, n_params) or None, default=None
            The inverse Fisher information matrix for each target. If None the
            class instance inverse Fisher information matrices are used
        ranges : list or None, default=None
            A list of arrays containing the gridpoints for the marginal
            distribution for each parameter. If None the class instance ranges
            are used determined by the prior range
        gridsize : list or None, default=None
            If using own `ranges` then the gridsize for these ranges must be
            passed (not checked)

        Returns
        -------
        list of lists:
            The 1D and 2D marginal distributions for each parameter (of pair)

        Todo
        ----
        Need to multiply the distribution by the prior to get the posterior
        Maybe move to TensorFlow probability?
        Make sure that using several Fisher estimates works
        """
        if parameter_estimates is None:
            parameter_estimates = self.parameter_estimates
        n_targets = parameter_estimates.shape[0]
        if invF is None:
            invF = self.invF
        if ranges is None:
            ranges = self.ranges
        if gridsize is None:
            gridsize = self.gridsize
        marginals = []
        for row in range(self.n_params):
            marginals.append([])
            for column in range(self.n_params):
                if column == row:
                    marginals[row].append(
                        jax.vmap(
                            lambda mean, _invF: norm.pdf(
                                ranges[column],
                                mean,
                                np.sqrt(_invF)))(
                                    parameter_estimates[:, column],
                                    invF[:, column, column]))
                elif column < row:
                    X, Y = np.meshgrid(ranges[row], ranges[column])
                    unravelled = np.vstack([X.ravel(), Y.ravel()]).T
                    marginals[row].append(
                        jax.vmap(
                            lambda mean, _invF: multivariate_normal.pdf(
                                unravelled,
                                mean,
                                _invF).reshape(
                                    ((gridsize[column], gridsize[row]))))(
                            parameter_estimates[:, [row, column]],
                            invF[:,
                                 [row, row, column, column],
                                 [row, column, row, column]].reshape(
                                (n_targets, 2, 2))))
        return marginals
