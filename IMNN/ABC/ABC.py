"""Approximate Bayesian computation with IMNN

This module provides the methods necessary to perform various ABC methods using
the IMNN.

TODO
____
This is a early update to make the ABC module work with TF2, this should be
properly ported at some point soon.
"""


__version__ = '0.2a1'
__author__ = "Tom Charnock"


import numpy as np
import tqdm
from ..utils.utils import utils
from .priors import TruncatedGaussian


class ABC():
    """Module containing ABC, PMC and gaussian approximation functions

    Attributes
    __________
    prior : class
        the truncated Gaussian priors to draw parameters values from
    fisher : ndarray
        Fisher information matrix calculated from last run summaries
    MLE : ndarray
        maximum likelihood estimate of the real data
    get_MLE : func
        get MLE from network
    simulator : func
        single input lambda function of the simulator
    n_params : int
        the number of parameters in the model
    ABC_dict : dict
        dictionary containing the parameters, summaries, distances and
        differences calculated during the ABC
    PMC_dict : dict
        dictionary containing the parameters, summaries, distances and
        differences calculated during the PMC
    total_draws : int
        the number of total draws from the proposal for the PMC
    """
    def __init__(self, real_data, prior, F, get_MLE, simulator,
                 seed, simulator_args):
        """Initialises the ABC class and calculates some useful values

        Parameters
        __________
        real_data : ndarray
            the observed data. in principle several observations can be passed
            at one time.
        prior : class
            the truncated Gaussian priors to draw parameters values from
        F : TF tensor float (n_params, n_params)
            approximate Fisher information to use for ABC
        get_MLE : func
            function for obtaining MLE from neural network
        simulator : func
            single input lambda function of the simulator
        seed : func
            function to set seed in the simulator
        simulator_args : dict
            simulator arguments to be passed to the simulator
        """
        self.prior = prior
        self.F = F.numpy()
        self.Finv = np.linalg.inv(F.numpy())
        self.get_MLE = get_MLE
        self.MLE = self.get_MLE(real_data).numpy()
        self.simulator = lambda x: simulator(x, seed, simulator_args)
        self.n_params = self.F.shape[0]
        self.ABC_dict = {
            "parameters": np.array([]).reshape((0, self.n_params)),
            "differences": np.array([]).reshape((0, self.n_params)),
            "MLE": np.array([]).reshape((0, self.n_params)),
            "distances": np.array([])}
        self.PMC_dict = {
            "parameters": np.array([]).reshape((0, self.n_params)),
            "MLE": np.array([]).reshape((0, self.n_params)),
            "differences": np.array([]).reshape((0, self.n_params)),
            "distances": np.array([])}
        self.total_draws = 0

    def ABC(self, draws, at_once=True, save_sims=None, return_dict=False,
            PMC=False):
        """Approximate Bayesian computation

        Here we draw some parameter values from the prior supplied to the class
        and generate simulations. We then use the IMNN to compress the sims
        into summaries and compare those to the summary of the observed data.

        All summaries are collected so that the acceptance epsilon can be
        modified at the users will.

        Parameters
        __________
        draws : int
            number of parameter draws to make, this number of simulations will
            be run.
        at_once : bool, optional
            whether to run all the simulations at once in parallel (the
            simulator function must handle this), or whether to run the
            simulations one at a time.
        save_sims : str, optional
            if the sims are costly it might be worth saving them. if a string
            is passed the sims will be saved as npz format from the arrays
            created.
        return_dict : bool, optional
            the ABC_dict attribute is normally updated, but the dictionary can
            be returned by the function. this is used by the PMC.
        PMC : bool, optional
            if this is true then the parameters are passed directly to ABC
            rather than being drawn in the ABC. this is used by the PMC.
        bar : func
            the function for the progress bar. this must be different depending
            on whether this is run in a notebook or not.
        parameters : ndarray
            the parameter values to run the simulations at
        sims : ndarray
            the simulations to compress to perform the ABC
        MLEs : ndarray
            the MLEs of the simulations from the IMNN
        differences : ndarray
            the difference between the observed data and all of the MLEs
        distances : ndarray
            the distance mesure describing how similar the observed MLE is
            to the MLEs of the simulations
        """
        if utils().isnotebook():
            bar = tqdm.tqdm_notebook
        else:
            bar = tqdm.tqdm
        if PMC:
            parameters = draws
            draws = parameters.shape[0]
        else:
            parameters = self.prior.draw(to_draw=draws)
        if at_once:
            sims = self.simulator(parameters)
            if save_sims is not None:
                np.savez(save_sims + ".npz", sims)
            MLEs = self.get_MLE(sims).numpy()
        else:
            MLEs = np.zeros([draws, self.n_params])
            for theta in bar(range(draws), desc="Simulations"):
                sim = self.simulator([parameters[theta]])
                if save_sims is not None:
                    np.savez(save_sims + "_" + str(theta), sim)
                MLEs[theta] = self.get_MLE([sim]).numpy()[0]
        differences = MLEs - self.MLE
        distances = np.sqrt(
            np.einsum(
                'ij,ij->i',
                differences,
                np.einsum(
                    'jk,ik->ij',
                    self.F,
                    differences)))

        if return_dict:
            ABC_dict = {"parameters": parameters,
                        "MLE": MLEs,
                        "differences": differences,
                        "distances": distances}
            return ABC_dict
        else:
            self.ABC_dict["parameters"] = np.concatenate(
                [self.ABC_dict["parameters"], parameters])
            self.ABC_dict["MLE"] = np.concatenate(
                [self.ABC_dict["MLE"], MLEs])
            self.ABC_dict["differences"] = np.concatenate(
                [self.ABC_dict["differences"], differences])
            self.ABC_dict["distances"] = np.concatenate(
                [self.ABC_dict["distances"], distances])

    def PMC(self, draws, posterior, criterion, at_once=True, save_sims=None,
            restart=False):
        """Population Monte Carlo

        This is the population Monte Carlo sequential ABC method, highly
        optimised for minimal numbers of drawsself.

        It works by first running an ABC and sorting the output distances,
        keeping the closest n parameters (where n is the number of samples to
        keep for the posterior) to get an initial proposal distribution.

        The proposal distribution is a Gaussian distribution with covariance
        given by weighted parameter values. Each iteration of draws moves 25%
        of the futhers samples until they are within the epsilon for that
        iteration. Once this is done the new weighting is calculated depending
        on the value of the new parameters and the new weighted covariance is
        calculated.

        Convergence is classified with a criterion which compares how many
        draws from the proposal distribution are needed to be accepted. When
        the number of draws is large then the posterior is fairly stable and
        can be trusted to really be approaching the true posterior distribution

        Parameters
        __________
        draws : int
            number of parameter draws from the prior to make on initialisation
        posterior : int
            number of samples to keep from the final provided posterior
        criterion : float
            the ratio to the number of samples to obtain in from the final
            posterior to the number of draws needed in an iteration.
        at_once : bool, optional
            whether to run all the simulations at once in parallel (the
            simulator function must handle this), or whether to run the
            simulations one at a time.
        save_sims : str, optional
            if the sims are costly it might be worth saving them. if a string
            is passed the sims will be saved as npz format from the arrays
            created.
        restart : bool, optional
            to restart the PMC from scratch set this value to true, otherwise
            the PMC just carries on from where it last left off. note that the
            weighting is reset, but it should level itself out after the first
            iteration.
        iteration : int
            counter for the number of iterations of the PMC to convergence.
        criterion_reached : float
            the value of the criterion after each iteration. once this reaches
            the supplied criterion value then the PMC stops.
        weighting : ndarray
            the weighting of the covariance for the proposal distribution.
        cov : ndarray
            the covariance of the parameter samples for the proposal
            distribution.
        epsilon : float
            the distance from the summary of the observed data where the
            samples are accepted.
        stored_move_ind : list
            the indices of the most distant parameter values which need to be
            moved during the PMC.
        move_ind : list
            the indices of the stored_move_ind which is decreased in size until
            all of the samples have been moved inside the epsilon.
        current_draws : int
            the number of draws taken when moving the samples in the population
        accepted_parameters : ndarray
            the parameter values which have been successfully moved during PMC.
        accepted_MLEs : ndarray
            the MLEs which have successfully moved closer than epsilon.
        accepted_differences : ndarray
            the difference between the observed data and all of the summaries.
        accepted_distances : ndarray
            the distance mesure describing how similar the observed summary is
            to the summaries of the simulations.
        proposed_parameters : ndarray
            the proposed parameter values to run simulations at to try and move
            closer to the true observation.
        temp_dictionary : dict
            dictionary output of the ABC with all summaries, parameters and
            distances.
        accept_index : list
            the indices of the accepted samples.
        reject_index : list
            the indices of the rejected samples.
        inv_cov : ndarray
            inverse covariance for the Gaussian proposal distribution.
        dist : ndarray
            the value of the proposal distribution at the accepted parameters.
        diff : ndarray
            difference between the accepted parameters and the parameter values
            from the previous iteration.
        """
        if self.total_draws == 0 or restart:
            self.PMC_dict = self.ABC(draws, at_once=at_once,
                                     save_sims=save_sims, return_dict=True)
            inds = self.PMC_dict["distances"].argsort()
            self.PMC_dict["parameters"] = self.PMC_dict[
                "parameters"][inds[:posterior]]
            self.PMC_dict["MLE"] = self.PMC_dict[
                "MLE"][inds[:posterior]]
            self.PMC_dict["differences"] = self.PMC_dict[
                "differences"][inds[:posterior]]
            self.PMC_dict["distances"] = self.PMC_dict[
                "distances"][inds[:posterior]]
            self.total_draws = 0

        weighting = np.ones(posterior) / posterior
        iteration = 0
        criterion_reached = 1e10
        while criterion < criterion_reached:
            draws = 0
            cov = np.cov(
                self.PMC_dict["parameters"],
                aweights=weighting,
                rowvar=False)
            if self.n_params == 1:
                cov = np.array([[cov]])
            epsilon = np.percentile(self.PMC_dict["distances"], 75)

            stored_move_ind = np.where(
                self.PMC_dict["distances"] >= epsilon)[0]
            move_ind = np.arange(stored_move_ind.shape[0])
            current_draws = move_ind.shape[0]
            accepted_parameters = np.zeros(
                (stored_move_ind.shape[0], self.n_params))
            accepted_distances = np.zeros((stored_move_ind.shape[0]))
            accepted_MLE = np.zeros(
                (stored_move_ind.shape[0], self.n_params))
            accepted_differences = np.zeros(
                (stored_move_ind.shape[0], self.n_params))
            while current_draws > 0:
                draws += current_draws
                proposed_parameters = TruncatedGaussian(
                    self.PMC_dict["parameters"][stored_move_ind[move_ind]],
                    cov,
                    self.prior.lower,
                    self.prior.upper).pmc_draw()
                temp_dictionary = self.ABC(
                    proposed_parameters,
                    at_once=at_once,
                    save_sims=save_sims,
                    return_dict=True, PMC=True)
                accept_index = np.where(
                    temp_dictionary["distances"] <= epsilon)[0]
                reject_index = np.where(
                    temp_dictionary["distances"] > epsilon)[0]
                accepted_parameters[move_ind[accept_index]] = \
                    temp_dictionary["parameters"][accept_index]
                accepted_distances[move_ind[accept_index]] = \
                    temp_dictionary["distances"][accept_index]
                accepted_MLE[move_ind[accept_index]] = \
                    temp_dictionary["MLE"][accept_index]
                accepted_differences[move_ind[accept_index]] = \
                    temp_dictionary["differences"][accept_index]
                move_ind = move_ind[reject_index]
                current_draws = move_ind.shape[0]

            inv_cov = np.linalg.inv(cov)
            dist = np.ones_like(weighting)
            diff = accepted_parameters \
                - self.PMC_dict["parameters"][stored_move_ind]
            dist[stored_move_ind] = np.exp(
                -0.5 * np.einsum(
                    "ij,ij->i",
                    np.einsum(
                        "ij,jk->ik",
                        diff,
                        inv_cov),
                    diff)) \
                / np.sqrt(2. * np.pi * np.linalg.det(cov))
            self.PMC_dict["parameters"][stored_move_ind] = accepted_parameters
            self.PMC_dict["distances"][stored_move_ind] = accepted_distances
            self.PMC_dict["MLE"][stored_move_ind] = accepted_MLE
            self.PMC_dict["differences"][stored_move_ind] = \
                accepted_differences
            weighting = self.prior.pdf(self.PMC_dict["parameters"]) \
                / np.sum(weighting * dist)
            criterion_reached = posterior / draws
            iteration += 1
            self.total_draws += draws
            print('iteration = ' + str(iteration)
                  + ', current criterion = ' + str(criterion_reached)
                  + ', total draws = ' + str(self.total_draws)
                  + ', ϵ = ' + str(epsilon) + '.', end='\r')

    def gaussian_approximation(self, gridsize=20):
        """Gaussian approximation of the posterior distribution

        Based on the maximum likelihood estimate of the observed data and
        the value of the inverse Fisher matrix we can make a Gaussian
        approximation to the posterior without making any extra simulations

        Parameters
        __________
        gridsize : int, optional
            the number of points to evaluate the distribution at in each
            dimension
        parameters : list of ndarray
            a list of each of the range of parameter values
        grid : ndarray
            the gridded parameter values to evaluate the distribution at
        dx : {list, float}
            first used to collect the step size in each parameter direction and
            then combined during the integral for normalisation
        ind : list of int
            indices for the gridded space to calculate the step size
        new_ind : list of int
            changed indices values for the gridded space to calculate step size
        span : tuple
            tuple to reshape the maximum likelihood estimate to correct shape
        diff : ndarray
            the difference between the maximum likelihood estimate and every
            value in the grid.
        fisher_string : str
            string describing dimension of the Fisher matrix to automatically
            do the einsum for any dimension
        grid_string : str
            string describing dimension grid to automatically do the einsum for
            any dimension
        first_string : str
            the einsum tensor notation for the second half of the distance
            calculation
        second_string : str
            the einsum tensor notation for the rest of the distance calculation
        posterior : ndarray
            the value of the approximate posterior evaluated at every gridpoint

        Returns
        _______
        ndarray
            the normalised gridded approximate posterior distribution.
        ndarray
            the grid at which the approximate posterior distribution is
            evaluated.
        """
        parameters = [np.linspace(
                self.prior.lower[i],
                self.prior.upper[i],
                gridsize)
            for i in range(self.n_params)]
        grid = np.array(np.meshgrid(*parameters, indexing="ij"))

        dx = []
        ind = np.zeros(self.n_params).astype(np.int).tolist()
        for i in range(self.n_params):
            new_ind = np.copy(ind).tolist()
            new_ind[i] = 1
            dx.append(grid[tuple([i] + new_ind)] - grid[tuple([i] + ind)])

        dx = np.prod(dx)
        span = tuple([...] + [np.newaxis for i in range(self.n_params)])
        diff = self.MLE[0][span] - grid

        fisher_string = "ij"
        grid_string = ""
        for i in range(self.n_params):
            grid_string += chr(ord("j") + i + 1)
        first_string = fisher_string + ",j" + grid_string + "->i" + grid_string
        second_string = "i" + grid_string + ",i" + grid_string + "->" \
            + grid_string[::-1]
        posterior = np.exp(-0.5 * np.einsum(
            second_string, diff, np.einsum(
                first_string, self.F, diff)) - 0.5
            * np.log(2. * np.pi * np.linalg.det(
                self.Finv)))
        return posterior / np.sum(dx * posterior), grid
