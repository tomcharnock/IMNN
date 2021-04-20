import matplotlib.pyplot as plt
import jax.numpy as np
from imnn.utils import get_gridsize


class LikelihoodFreeInference:
    def __init__(self, prior, gridsize=100, verbose=True):
        self.verbose = verbose
        self.prior = prior
        self.n_params = len(self.prior.event_shape)
        self.gridsize = get_gridsize(gridsize, self.n_params)
        self.ranges = [
            np.linspace(
                self.prior.low[i],
                self.prior.high[i],
                self.gridsize[i])
            for i in range(self.n_params)]
        self.marginals = None
        self.n_targets = None

    def get_levels(self, marginal, ranges, levels=[0.68, 0.95]):
        domain_volume = 1
        for i in range(len(ranges)):
            domain_volume *= ranges[i][1] - ranges[i][0]
        sorted_marginal = np.sort(marginal.flatten())[::-1]
        cdf = np.cumsum(sorted_marginal / sorted_marginal.sum())
        value = []
        for level in levels[::-1]:
            this_value = sorted_marginal[np.argmin(np.abs(cdf - level))]
            if len(value) == 0:
                value.append(this_value)
            elif this_value <= value[-1]:
                break
            else:
                value.append(this_value)
        return value

    def setup_plot(self, ax=None, ranges=None, axis_labels=None,
                   figsize=(10, 10), format=False):
        rows = len(ranges)
        columns = len(ranges)
        if ax is None:
            fig, ax = plt.subplots(rows, columns, figsize=figsize)
            plt.subplots_adjust(wspace=0.1, hspace=0.1)
        elif not format:
            return ax
        for column in range(columns):
            for row in range(rows):
                if column > row:
                    ax[row, column].axis("off")
                else:
                    if row > 0:
                        ax[row, column].set_ylim(
                            ranges[row][0],
                            ranges[row][-1])
                        if (column == 0) and (axis_labels is not None):
                            ax[row, column].set_ylabel(axis_labels[row])
                    else:
                        ax[row, column].set_yticks([])
                    if row < rows - 1:
                        ax[row, column].set_xticks([])
                    if column > 0:
                        ax[row, column].set_yticks([])
                    if column < columns - 1:
                        ax[row, column].set_xlim(
                            ranges[column][0],
                            ranges[column][-1])
                        if (row == rows - 1) and (axis_labels is not None):
                            ax[row, column].set_xlabel(axis_labels[column])
                    else:
                        ax[row, column].set_xticks([])
        return ax

    def scatter_plot_(self, ax=None, ranges=None, points=None, label=None,
                      axis_labels=None, colours=None, hist=True, s=5, alpha=1.,
                      figsize=(10, 10), linestyle="solid", target=None,
                      format=False, ncol=2, bbox_to_anchor=(0.0, 1.0)):
        targets, n_targets = self.target_choice(target)
        if colours is None:
            colours = [f"C{i}" for i in range(n_targets)]
        if type(colours) is str:
            colours = [colours for i in range(n_targets)]
        if ranges is None:
            ranges = self.ranges
        if label is None:
            label = ""
        rows = len(ranges)
        columns = len(ranges)
        ax = self.setup_plot(ax=ax, ranges=ranges, axis_labels=axis_labels,
                             figsize=figsize, format=format)
        for column in range(columns):
            for row in range(rows):
                for i, target in enumerate(targets):
                    if (column == row) and hist:
                        if column < columns - 1:
                            ax[row, column].hist(
                                points[target][:, row],
                                bins=ranges[row],
                                color=colours[i],
                                linestyle=linestyle,
                                density=True,
                                histtype="step")
                        else:
                            ax[row, column].hist(
                                points[target][:, row],
                                bins=ranges[column],
                                color=colours[i],
                                linestyle=linestyle,
                                density=True,
                                histtype="step",
                                orientation="horizontal")

                    elif column < row:
                        ax[row, column].scatter(
                            points[target][:, column],
                            points[target][:, row],
                            s=s,
                            color=colours[i],
                            alpha=alpha,
                            label=label + f" Target {target+1}")
                        h, l = ax[row, column].get_legend_handles_labels()
        if label != "":
            ax[0, 0].legend(h, l, frameon=False, bbox_to_anchor=bbox_to_anchor,
                            ncol=ncol)
        return ax

    def scatter_plot(self, ax=None, ranges=None, points=None, label=None,
                     axis_labels=None, colours=None, hist=True, s=5, alpha=1.,
                     figsize=(10, 10), linestyle="solid", target=None,
                     format=False, ncol=2, bbox_to_anchor=(0.0, 1.0)):
        if ranges is None:
            raise ValueError("`ranges` must be provided")
        if points is None:
            raise ValueError("`points` to scatter must be provided")
        return self.scatter_plot_(
            ax=ax, ranges=ranges, points=points, label=label,
            axis_labels=axis_labels, colours=colours, hist=hist, s=s,
            alpha=alpha, figsize=figsize, linestyle=linestyle, target=target,
            format=format, ncol=ncol, bbox_to_anchor=bbox_to_anchor)

    def marginal_plot(self, ax=None, ranges=None, marginals=None, label=None,
                      axis_labels=None, levels=None, linestyle="solid",
                      colours=None, target=None, format=False, ncol=2,
                      bbox_to_anchor=(1.0, 1.0)):
        targets, n_targets = self.target_choice(target)
        if (marginals is None) and (self.marginals is None):
            if self.verbose:
                print("Need to provide `marginal` or run `get_marginals()`")
        elif marginals is None:
            marginals = self.marginals
        if levels is None:
            levels = [0.68, 0.95]
        if colours is None:
            colours = [f"C{i}" for i in range(n_targets)]
        if type(colours) is str:
            colours = [colours for i in range(n_targets)]
        if ranges is None:
            ranges = self.ranges
        if label is None:
            label = ""
        rows = len(ranges)
        columns = len(ranges)
        ax = self.setup_plot(ax=ax, ranges=ranges, axis_labels=axis_labels,
                             format=format)
        for column in range(columns):
            for row in range(rows):
                for i, target in enumerate(targets):
                    if column == row:
                        if column < columns - 1:
                            ax[row, column].plot(
                                ranges[row],
                                marginals[row][column][target],
                                color=colours[i],
                                linestyle=linestyle,
                                label=label)
                        else:
                            ax[row, column].plot(
                                marginals[row][column][target],
                                ranges[row],
                                color=colours[i],
                                linestyle=linestyle)
                    elif column < row:
                        ax[row, column].contour(
                            ranges[column],
                            ranges[row],
                            marginals[row][column][target].T,
                            colors=colours[i],
                            linestyles=[linestyle],
                            levels=self.get_levels(
                                marginals[row][column][target],
                                [ranges[column], ranges[row]],
                                levels=levels))
        if label != "":
            ax[0, 0].legend(
                frameon=False,
                bbox_to_anchor=bbox_to_anchor,
                ncol=ncol)
        return ax

    def target_choice(self, target):
        if target is None:
            targets = range(self.n_targets)
            n_targets = self.n_targets
        elif type(target) == list:
            targets = target
            n_targets = len(target)
        else:
            targets = [target]
            n_targets = 1
        return targets, n_targets
