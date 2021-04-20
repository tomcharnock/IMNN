import jax
import jax.numpy as np
from jax.scipy.stats import norm, multivariate_normal
from imnn.lfi import LikelihoodFreeInference


class GaussianApproximation(LikelihoodFreeInference):
    def __init__(self, target_summaries, invF, prior, gridsize=100):
        super().__init__(
            prior=prior,
            gridsize=gridsize)
        if len(target_summaries.shape) == 0:
            target_summaries = np.expand_dims(target_summaries, 0)
        if len(target_summaries.shape) == 1:
            target_summaries = np.expand_dims(target_summaries, 0)
        self.target_summaries = target_summaries
        self.n_targets = self.target_summaries.shape[0]
        self.n_summaries = self.target_summaries.shape[-1]
        self.invF = invF
        self.marginals = self.get_marginals()

    def get_marginals(self, target_summaries=None, invF=None, ranges=None,
                      gridsize=None):
        if target_summaries is None:
            target_summaries = self.target_summaries
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
                            lambda mean: norm.pdf(
                                ranges[column],
                                mean,
                                np.sqrt(invF[column, column])))(
                                    target_summaries[:, column]))
                elif column < row:
                    X, Y = np.meshgrid(ranges[row], ranges[column])
                    unravelled = np.vstack([X.ravel(), Y.ravel()]).T
                    marginals[row].append(
                        jax.vmap(
                            lambda mean: multivariate_normal.pdf(
                                unravelled,
                                mean,
                                invF[
                                    [row, row, column, column],
                                    [row, column, row, column]].reshape(
                                        (2, 2))).reshape(
                                            ((gridsize[column],
                                              gridsize[row]))))(
                            target_summaries[:, [row, column]]))
        return marginals
