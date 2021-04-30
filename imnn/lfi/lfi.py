import matplotlib.pyplot as plt
import jax.numpy as np


class LikelihoodFreeInference:
    """Base class (some functionality) for likelihood free inference methods

    Mostly used for plotting

    Parameters
    ----------
    gridsize : list
        The number of grid points to evaluate the marginal distribution on for
        each parameter
    ranges : list
        A list of arrays containing the gridpoints for the marginal
        distribution for each parameter
    marginal : list of lists
        A list of rows and columns of marginal distributions of a corner plot

    Methods
    -------
    prior
        A prior distribution which can be evaluated and sampled from (should
        also contain a ``low`` and a ``high`` attribute with appropriate
        ranges)

    Todo
    ----
    pytests need writing
    """
    def __init__(self, prior, gridsize=100, marginals=None):
        """Constructor method

        Parameters
        ----------
        prior: fn
            A prior distribution which can be evaluated and sampled from
            (should also contain a ``low`` and a ``high`` attribute with
            appropriate ranges)
        gridsize : int or list, default=100
            The number of grid points to evaluate the marginal distribution on
            for every parameter (int) or each parameter (list)
        marginals : float(n_targets, gridsize*n_params) or None, default=None
            The full distribution evaluated on a grid to put into marginal list
        """
        self.prior = prior
        try:
            self.n_params = self.prior.event_shape[0]
        except Exception:
            raise ValueError(
                "`prior` has no event_shape - this should be `n_params`")
        if not hasattr(self.prior, "low"):
            raise ValueError(
                "`prior` must have (or be given by assignment) a `low` " +
                "attribute describing the minimum allowed value for each " +
                "parameter value")
        if not hasattr(self.prior, "low"):
            raise ValueError(
                "`prior` must have (or be given by assignment) a `low` " +
                "attribute describing the minimum allowed value for each " +
                "parameter value")
        self.gridsize = self.get_gridsize(gridsize, self.n_params)
        self.ranges = [
            np.linspace(
                self.prior.low[i],
                self.prior.high[i],
                self.gridsize[i])
            for i in range(self.n_params)]
        self.marginals = self.put_marginals(marginals)

    def get_gridsize(self, gridsize, size):
        """Propogates gridpoints per parameter into a list if int provided

        Parameters
        ----------
        gridsize : int or list
            The number of grid points to evaluate the marginal distribution on
            for every parameter (int) or each parameter (list)
        size : int
            Number of parameters describing size of gridsize list

        Returns
        -------
        list:
            The number of gridpoints to evaluate marginals for each parameter

        Raises
        ------
        ValueError
            If list passed is wrong length
        TypeError
            If gridsize is not int or list of correct length
        """
        if isinstance(gridsize, int):
            gridsize = [gridsize for i in range(size)]
        elif isinstance(gridsize, list):
            if len(gridsize) == size:
                gridsize = gridsize
            else:
                raise ValueError(
                    f"`gridsize` is a list of length {len(gridsize)} but " +
                    f"`shape` determined by `input` is {size}")
        else:
            raise TypeError("`gridsize` is not a list or an integer")
        return gridsize

    def get_levels(self, marginal, ranges, levels=[0.68, 0.95]):
        """ Used for approximately calculating confidence region isocontours

        To calculate the values of the marginal distribution whose isocontour
        contains approximately `levels` specified fraction of samples drawn
        from the distribution the marginal distribution is sorted by value and
        normalised to one. If the distribution does is significantly non-zero
        outside of the range then this will cause large biases! The cumulative
        sum of the sorted values are then calculated and the value at which
        the index of this normalised cumulative distribution is closest to the
        desired level is used to return the value of the original marginal
        distribution.

        Parameters
        ----------
        marginal : float(gridsize*n_param)
            The gridded marginal distribution to find the isocontour values of
        ranges : list
            List of the grid points for each parameter
        levels : list, default=[0.68, 0.95]
            The fraction describing the percentage of samples inside the
            isocontour

        Returns
        -------
        list:
            The values of the isocontours of the marginal distributions
        """
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
        """Builds corner plot

        ax : list of matplotlib axes or None, default=None
            If ax is None a new figure and axes are created to make a corner
            plot otherwise a set of axes are formatted correctly for a corner
            plot. If existing ax is not correctly shaped there will be an error
        ranges : list, default=None
            The list of ranges to set the number of rows and columns (if this
            is None there will be an error)
        axis_labels : list of str or None, default=None
            A list of names for each parameter, no axis labels if None
        figsize : tuple, default=(10, 10)
            The size (in inches) to create a figure (if ax is None)
        format : bool, default=False
            If formatting is not needed

        Returns
        -------
        axes object:
            The formatted matplotlib axes
        """
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

    def _scatter_plot(self, ax=None, ranges=None, points=None, label=None,
                      axis_labels=None, colours=None, hist=True, s=5.,
                      alpha=1., figsize=(10, 10), linestyle="solid",
                      target=None, format=False, ncol=2,
                      bbox_to_anchor=(0.0, 1.0)):
        """Plotter for scatter plots

        Plots scatter plots for points (parameters or summaries) in 2D subplots
        and the histogram of points in the 1D diagonal subplots.

        Parameters
        ----------
        ax : list of matplotlib axes or None, default=None
            If ax is None a new figure and axes are created to make a corner
            plot otherwise a set of axes are formatted correctly for a corner
            plot. If existing ax is not correctly shaped there will be an error
        ranges : list, default=None
            The list of ranges to set the number of rows and columns (if this
            is None the ranges set on initialisation will be used)
        points : float(n_targets, n_points, {n_params} or {n_summaries})
            The points to scatter plot
        label : str or None, default=None
            Name to be used in the legend
        axis_labels : list of str or None, default=None
            A list of names for each parameter, no axis labels if None
        colours : str or list or None, default=None
            The colour or list of colours to use for the different targets, if
            None then the normal matplotlib colours are used
        hist : bool, default=True
            Whether or not to plot histograms on the diagonal of the corner
            plot
        s : float, default=5.
            The size of the marker points in the scatter plot
        alpha : float, default=1.
            The amount of alpha colour for the marker points
        figsize : tuple, default=(10, 10)
            The size (in inches) to create a figure (if ax is None)
        linestyle : str, default="solid"
            Linestyle for the histograms
        target : None or int or list, default=None
            The indices to choose which target's points are plotted
        format : bool, default=False
            If formatting is not needed
        ncols : int, default=2
            Number of columns for the legend
        bbox_to_anchor : tuple, default=(0.0, 1.0)
            Position to fix the legend to

        Returns
        -------
        axes object:
            The scatter plot axes

        Raises
        ------
        ValueError
            if colour input is not correct

        Todo
        ----
        There are many extra matplotlib parameters which could be passed,
        although this is not done because the axis is returned which can then
        be manipulated.
        """
        if ranges is None:
            raise ValueError("`ranges` must be provided")
        if points is None:
            raise ValueError("`points` to scatter must be provided")
        targets, n_targets = self.target_choice(target)
        if colours is None:
            colours = [f"C{i}" for i in range(n_targets)]
        elif isinstance(colours, str):
            colours = [colours for i in range(n_targets)]
        elif isinstance(colours, list):
            pass
        else:
            raise ValueError(
                "`colours` must be None, a color as a string or a list of " +
                "colours")
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
                        h, lab = ax[row, column].get_legend_handles_labels()
        if label != "":
            ax[0, 0].legend(h, lab, frameon=False,
                            bbox_to_anchor=bbox_to_anchor, ncol=ncol)
        return ax

    def marginal_plot(self, ax=None, ranges=None, marginals=None, known=None,
                      label=None, axis_labels=None, levels=None,
                      linestyle="solid", colours=None, target=None,
                      format=False, ncol=2, bbox_to_anchor=(1.0, 1.0)):
        """Plots the marginal distribution corner plots

        Plots the 68% and 95% (approximate) confidence contours for each 2D
        parameter pair and the 1D densities as a corner plot.

        Parameters
        ----------
        ax : list of matplotlib axes or None, default=None
            If ax is None a new figure and axes are created to make a corner
            plot otherwise a set of axes are formatted correctly for a corner
            plot. If existing ax is not correctly shaped there will be an error
        ranges : list, default=None
            The list of ranges to set the number of rows and columns (if this
            is None the ranges set on initialisation will be used)
        marginals : list of lists (n_params * n_params) or None, default=None
            The marginal distributions for every parameter and every 2D
            combination. If None the marginals defined in the class are used
            (if they exist)
        known : float(n_params,)
            Values to plot known parameter value lines at
        label : str or None, default=None
            Name to be used in the legend
        axis_labels : list of str or None, default=None
            A list of names for each parameter, no axis labels if None
        linestyle : str, default="solid"
            Linestyle for the histograms
        colours : str or list or None, default=None
            The colour or list of colours to use for the different targets, if
            None then the normal matplotlib colours are used
        target : None or int or list, default=None
            The indices to choose which target's points are plotted
        format : bool, default=False
            If formatting is not needed
        ncols : int, default=2
            Number of columns for the legend
        bbox_to_anchor : tuple, default=(0.0, 1.0)
            Position to fix the legend to

        Returns
        -------
        axes object:
            The scatter plot axes

        Raises
        ------
        ValueError
            if `marginals` is not provided
        TypeError
            if `marginals` is not a list

        Todo
        ----
        There are many extra matplotlib parameters which could be passed,
        although this is not done because the axis is returned which can then
        be manipulated.
        """
        targets, n_targets = self.target_choice(target)
        if (marginals is None) and (self.marginals is None):
            raise ValueError(
                "Need to provide `marginal` or run `get_marginals()`")
        elif marginals is None:
            marginals = self.marginals
        elif not isinstance(marginals, list):
            raise TypeError
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
                            if known is not None:
                                ax[row, column].axvline(
                                    known[column],
                                    color="black",
                                    linestyle="dashed")
                        else:
                            ax[row, column].plot(
                                marginals[row][column][target],
                                ranges[row],
                                color=colours[i],
                                linestyle=linestyle)
                            if known is not None:
                                ax[row, column].axhline(
                                    known[column],
                                    color="black",
                                    linestyle="dashed")
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
                        if known is not None:
                            ax[row, column].axvline(
                                known[column],
                                color="black",
                                linestyle="dashed")
                            ax[row, column].axhline(
                                known[row],
                                color="black",
                                linestyle="dashed")
        if label != "":
            ax[0, 0].legend(
                frameon=False,
                bbox_to_anchor=bbox_to_anchor,
                ncol=ncol)
        return ax

    def put_marginals(self, marginals):
        """Creates list of 1D and 2D marginal distributions ready for plotting

        The marginal distribution lists from full distribution array. For every
        parameter the full distribution is summed over every other parameter to
        get the 1D marginals and for every combination the 2D marginals are
        calculated by summing over the remaining parameters. The list is made
        up of a list of n_params lists which contain n_columns number of
        objects.

        Parameters
        ----------
        marginals : float(n_targets, gridsize*n_params)
            The full distribution from which to calculate the marginal
            distributions

        Returns
        -------
        list of lists:
            The 1D and 2D marginal distributions for each parameter (of pair)
        """
        if marginals is None:
            return None
        _marginals = []
        for row in range(self.n_params):
            _marginals.append([])
            for column in range(self.n_params):
                if column == row:
                    _marginals[row].append(
                        marginals.sum(
                            tuple(i + 1 for i in range(self.n_params)
                                  if i != row)))
                    _marginals[row][-1] /= np.expand_dims(
                        np.sum(_marginals[row][-1], 1)
                        * (self.ranges[row][1] - self.ranges[row][0]),
                        1)
                elif column < row:
                    _marginals[row].append(
                        marginals.sum(
                            tuple(i + 1 for i in range(self.n_params)
                                  if (i != row) and (i != column))))
        self.marginals = _marginals
        return _marginals

    def target_choice(self, target):
        """Returns the indices of targets to plot and the number of targets

        Parameters
        ----------
        target : None or int or list
            The indices of the targets to plot. If None then ``n_targets`` from
            the class instance is used

        Returns
        -------
        list:
            The indices of the targets to be plotted
        int:
            The number of targets to be plotted
        """
        if target is None:
            if self.n_targets is None:
                raise ValueError("``n_targets`` class attribute not defined")
            targets = range(self.n_targets)
            n_targets = self.n_targets
        elif isinstance(target, list):
            targets = target
            n_targets = len(target)
        elif isinstance(targets, int):
            targets = [target]
            n_targets = 1
        else:
            raise ValueError(
                "`target` must be None (to use targets defined in " +
                "initialisation), a list of indices for the desired targets " +
                "to be plotted or an integer for the single target to be " +
                "plotted.")
        return targets, n_targets
