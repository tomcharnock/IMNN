import jax
import jax.numpy as np
import tqdm.auto as tqdm
from jax.tree_util import tree_flatten, tree_unflatten
from functools import partial
from imnn.utils.utils import _check_host, _check_devices, _check_boolean, \
    _check_type, _check_splitting, _check_input


class _AggregatedIMNN:
    """Manual aggregation of gradients for the IMNN parent class

    This class defines the overriding fitting functions for
    :func:`~imnn.imnn._imnn._IMNN` which allows gradients to be aggregated
    manually. This is necessary if networks or input data are extremely large
    (or if the number of simulations necessary to estimate the covariance of
    network outputs, ``n_s``, is very large) since all operations may not fit
    in memory.

    The aggregation is done by calculating ``n_per_device`` network outputs at
    once on each available ``jax.device()`` and then scanning over all ``n_s``
    inputs and ``n_d`` simulations necessary to calculate the derivative of the
    mean of the network outputs with respect to the model parameters. This
    gives a set of summaries and derivatives from which the loss function

    .. math::
        \\Lambda = -\\log|\\bf{F}| + r(\\Lambda_2) \\Lambda_2

    (See :doc:`/pages/details`) can be calculated and its gradient with respect
    to these summaries, :math:`\\frac{\\partial\\Lambda}{\\partial x_i^j}` and
    derivatives :math:`\\frac{\\partial\\Lambda}{\\partial\\partial{x_i^j}/
    \\partial\\theta_\\alpha}` calculated, where :math:`i` labels the network
    output and :math:`j` labels the simulation. Note that these are small in
    comparison to the gradient with respect to the network parameters since
    their sizes are ``n_s * n_summaries`` and ``n_d * n_summaries * n_params``
    respectively. Once :math:`\\frac{\\partial\\Lambda}{\\partial{x_i^j}}` and
    :math:`\\frac{\\partial\\Lambda}{\\partial\\partial{x_i^j}/\\partial
    \\theta_\\alpha}` are calculated then gradients of the network outputs with
    respect to the network parameters
    :math:`\\frac{\\partial{x_i^j}}{\\partial{w_{ab}^l}}` and
    :math:`\\frac{\\partial\\partial{x_i^j}/\\partial\\theta_\\alpha}
    {\\partial{w_{ab}^l}}` are calculated and the chain rule is used to get

    .. math::
        \\frac{\\partial\\Lambda}{\\partial{w_{ab}^l}} = \\frac{\\partial
        \\Lambda}{\\partial{x_i^j}}\\frac{\\partial{x_i^j}}
        {\\partial{w_{ab}^l}} + \\frac{\\partial\\Lambda}
        {\\partial\\partial{x_i^j}/\\partial\\theta_\\alpha}
        \\frac{\\partial\\partial{x_i^j}/\\partial\\theta_\\alpha}
        {\\partial{w_{ab}^l}}

    Note that we keep the memory use low because only ``n_per_device``
    simulations are handled at once before being summed into a single gradient
    list on each device.

    ``n_per_devices`` should be as large as possible to get the best
    performance. If everything will fit in memory then this class should be
    avoided.

    The AttributedIMNN class doesn't directly inherit from
    :func:`~imnn.imnn._imnn._IMNN`, but is meant to be built within a child
    class of it. For this reason there are attributes which are not explicitly
    set here, but are used within the module. These will be noted in
    Other Parameters below.

    Parameters
    ----------
    host: jax.device
        The main device where the Fisher information calculation is performed
    devices: list
        A list of the available jax devices (from ``jax.devices()``)
    n_devices: int
        Number of devices to aggregated calculation over
    n_per_device: int
        Number of simulations to handle at once, this should be as large as
        possible without letting the memory overflow for the best performance

    Methods
    -------
    model:
        Neural network as a function of network parameters and inputs
    _get_parameters:
        Function which extracts the network parameters from the state
    _model_initialiser:
        Function to initialise neural network weights from RNG and shape tuple
    _opt_initialiser:
        Function which generates the optimiser state from network parameters
    _update:
        Function which updates the state from a gradient
    batch_summaries:
        Jitted function to calculate summaries on each XLA device
    batch_summaries_with_derivatives:
        Jitted function to calculate summaries from derivative on each device
    batch_gradients:
        Jitted function to calculate gradient on each XLA device
    batch_gradients_with_derivatives:
        Jitted function to calculate gradient from derivative on eachdevice
    """
    def __init__(self, host, devices, n_per_device):
        """Constructor method

        Parameters
        ----------
        host: jax.device
            The main device where the Fisher calculation is performed
        devices: list
            A list of the available jax devices (from ``jax.devices()``)
        n_per_device: int
            Number of simulations to handle at once, this should be as large as
            possible without letting the memory overflow for the best
            performance
        """
        self.host = _check_host(host)
        self._set_devices(devices, n_per_device)
        self._set_batch_functions()
        self._set_shapes()

    def _set_devices(self, devices, n_per_device):
        """Checks that devices exist and that reshaping onto devices can occur

        Due to the aggregation then balanced splits must be made between the
        different devices and so these are checked.

        Parameters
        ----------
        devices: list
            A list of the available jax devices (from ``jax.devices()``)
        n_per_device: int
            Number of simulations to handle at once, this should be as large as
            possible without letting the memory overflow for the best
            performance

        Raises
        ------
        ValueError
            If ``devices`` or ``n_per_device`` are None
        ValueError
            If balanced splitting cannot be done
        TypeError
            If ``devices`` is not a list and if ``n_per_device`` is not an int
        """
        self.devices = _check_devices(devices)
        self.n_devices = len(self.devices)
        self.n_per_device = _check_type(n_per_device, int, "n_per_device")
        if self.n_s == self.n_d:
            _check_splitting(self.n_s, "n_s and n_d", self.n_devices,
                             self.n_per_device)
        else:
            _check_splitting(
                self.n_s, "n_s", self.n_devices, self.n_per_device)
            _check_splitting(
                self.n_d, "n_d", self.n_devices, self.n_per_device)

    def _set_batch_functions(self):
        """ Creates jitted functions placed on desired XLA devices

        For each set of summaries to correctly be calculated on a particular
        device we predefine the jitted functions on each of these devices
        """
        self.batch_summaries = [
            jax.jit(
                partial(
                    self._get_batch_summaries,
                    θ=self.θ_fid,
                    gradient=False,
                    derivative=False),
                device=device)
            for device in self.devices]
        self.batch_summaries_with_derivatives = [
            jax.jit(
                partial(
                    self._get_batch_summaries,
                    θ=self.θ_fid,
                    gradient=False,
                    derivative=True),
                device=device)
            for device in self.devices]
        self.batch_gradients = [
            jax.jit(
                partial(
                    self._get_batch_summaries,
                    θ=self.θ_fid,
                    gradient=True,
                    derivative=False),
                device=device)
            for device in self.devices]
        self.batch_gradients_with_derivatives = [
            jax.jit(
                partial(
                    self._get_batch_summaries,
                    θ=self.θ_fid,
                    gradient=True,
                    derivative=True),
                device=device)
            for device in self.devices]

    def _set_shapes(self):
        """Calculates the shapes for batching over different devices

        Not implemented

        Raises
        ------
        ValueError
            Not implemented in _AggregatedIMNN
        """
        raise ValueError("`_set_shapes` not implemented in `_AggregatedIMNN`")

    def fit(self, λ, ϵ, rng=None, patience=100, min_iterations=100,
            max_iterations=int(1e5), print_rate=None, best=True):
        """Fitting routine for the IMNN

        Parameters
        ----------
        λ : float
            Coupling strength of the regularisation
        ϵ : float
            Closeness criterion describing how close to the 1 the determinant
            of the covariance (and inverse covariance) of the network outputs
            is desired to be
        rng : int(2,) or None, default=None
            Stateless random number generator
        patience : int, default=10
            Number of iterations where there is no increase in the value of the
            determinant of the Fisher information matrix, used for early
            stopping
        min_iterations : int, default=100
            Number of iterations that should be run before considering early
            stopping using the patience counter
        max_iterations : int, default=int(1e5)
            Maximum number of iterations to run the fitting procedure for
        print_rate : int or None, default=None,
            Number of iterations before updating the progress bar whilst
            fitting. There is a performance hit from updating the progress bar
            more often and there is a large performance hit from using the
            progress bar at all. (Possible ``RET_CHECK`` failure if
            ``print_rate`` is not ``None`` when using GPUs).
            For this reason it is set to None as default
        best : bool, default=True
            Whether to set the network parameter attribute ``self.w`` to the
            parameter values that obtained the maximum determinant of
            the Fisher information matrix or the parameter values at the final
            iteration of fitting

        Example
        -------

        We are going to summarise the mean and variance of some random Gaussian
        noise with 10 data points per example using an AggregatedSimulatorIMNN.
        In this case we are going to generate the simulations on-the-fly with a
        simulator written in jax (from the examples directory). These
        simulations will be generated on-the-fly and passed through the network
        on each of the GPUs in ``jax.devices("gpu")`` and we will make 100
        simulations on each device at a time. The main computation will be done
        on the CPU. We will use 1000 simulations to estimate the covariance of
        the network outputs and the derivative of the mean of the network
        outputs with respect to the model parameters (Gaussian mean and
        variance) and generate the simulations at a fiducial μ=0 and Σ=1. The
        network will be a stax model with hidden layers of ``[128, 128, 128]``
        activated with leaky relu and outputting 2 summaries. Optimisation will
        be via Adam with a step size of ``1e-3``. Rather arbitrarily we'll set
        the regularisation strength and covariance identity constraint to λ=10
        and ϵ=0.1 (these are relatively unimportant for such an easy model).

        .. code-block:: python

            import jax
            import jax.numpy as np
            from jax.experimental import stax, optimizers
            from imnn import AggregatedSimulatorIMNN

            rng = jax.random.PRNGKey(0)

            n_s = 1000
            n_d = 1000
            n_params = 2
            n_summaries = 2
            input_shape = (10,)
            θ_fid = np.array([0., 1.])

            def simulator(rng, θ):
                return θ[0] + jax.random.normal(
                    rng, shape=input_shape) * np.sqrt(θ[1])

            model = stax.serial(
                stax.Dense(128),
                stax.LeakyRelu,
                stax.Dense(128),
                stax.LeakyRelu,
                stax.Dense(128),
                stax.LeakyRelu,
                stax.Dense(n_summaries))
            optimiser = optimizers.adam(step_size=1e-3)

            λ = 10.
            ϵ = 0.1

            model_key, fit_key = jax.random.split(rng)

            host = jax.devices("cpu")[0]
            devices = jax.devices("gpu")

            n_per_device = 100

            imnn = AggregatedSimulatorIMNN(
                n_s=n_s, n_d=n_d, n_params=n_params, n_summaries=n_summaries,
                input_shape=input_shape, θ_fid=θ_fid, model=model,
                optimiser=optimiser, key_or_state=model_key,
                simulator=simulator, host=host, devices=devices,
                n_per_device=n_per_device)

            imnn.fit(λ, ϵ, rng=fit_key, min_iterations=1000, patience=250,
                     print_rate=None)


        Notes
        -----
        A minimum number of interations should be be run before stopping based
        on a maximum determinant of the Fisher information achieved since the
        loss function has dual objectives. Since the determinant of the
        covariance of the network outputs is forced to 1 quickly, this can be
        at the detriment to the value of the determinant of the Fisher
        information matrix early in the fitting procedure. For this reason
        starting early stopping after the covariance has converged is advised.
        This is not currently implemented but could be considered in the
        future.

        The best fit network parameter values are probably not the most
        representative set of parameters when simulating on-the-fly since there
        is a high chance of a statistically overly-informative set of data
        being generated. Instead, if using
        :func:`~imnn.AggregatedSimulatorIMNN.fit()` consider using
        ``best=False`` which sets ``self.w=self.final_w`` which are the network
        parameter values obtained in the last iteration. Also consider using a
        larger ``patience`` value if using :func:`~imnn.SimulatorIMNN.fit()`
        to overcome the fact that a flukish high value for the determinant
        might have been obtained due to the realisation of the dataset.

        Raises
        ------
        TypeError
            If any input has the wrong type
        ValueError
            If any input (except ``rng``) are ``None``
        ValueError
            If ``rng`` has the wrong shape
        ValueError
            If ``rng`` is ``None`` but simulating on-the-fly

        Methods
        -------
        get_keys_and_params:
            Jitted collection of parameters and random numbers
        calculate_loss:
            Returns the jitted gradient of the loss function wrt summaries
        validation_loss:
            Jitted loss and auxillary statistics from validation set

        Todo
        ----
        - ``rng`` is currently only used for on-the-fly simulation but could
          easily be updated to allow for stochastic models
        - Automatic detection of convergence based on value ``r`` when early
          stopping can be started
        """
        @jax.jit
        def get_keys_and_params(rng, state):
            """Jitted collection of parameters and random numbers

            Parameters
            ----------
            rng : int(2,) or None, default=None
                Stateless random number generator
            state : :obj:state
                The optimiser state used for updating the network parameters
                and optimisation algorithm

            Returns
            -------
            int(2,) or None, default=None:
                Stateless random number generator
            int(2,) or None, default=None:
                Stateless random number generator for training
            int(2,) or None, default=None:
                Stateless random number generator for validation
            list:
                Network parameter values
            """
            rng, training_key, validation_key = self._get_fitting_keys(rng)
            w = self._get_parameters(state)
            return rng, training_key, validation_key, w

        @jax.jit
        @partial(jax.grad, argnums=(0, 1), has_aux=True)
        def calculate_loss(summaries, summary_derivatives):
            """Returns the jitted gradient of the loss function wrt summaries

            Used to calculate the gradient of the loss function wrt summaries
            and derivatives of the summaries with respect to model parameters
            which will be used to calculate the aggregated gradient of the
            Fisher information with respect to the network parameters via the
            chain rule.

            Parameters
            ----------
            summaries : float(n_s, n_summaries)
                The network outputs
            summary_derivatives : float(n_d, n_summaries, n_params)
                The derivative of the network outputs wrt the model parameters

            Returns
            -------
            tuple:
                Gradient of the loss function with respect to network outputs
                and their derivatives with respect to physical model parameters
            tuple:
                Fitting statistics calculated on a single iteration
                    - **F** *(float(n_params, n_params))* -- Fisher information
                      matrix
                    - **C** *(float(n_summaries, n_summaries))* -- covariance
                      of network outputs
                    - **invC** *(float(n_summaries, n_summaries))* -- inverse
                      covariance of network outputs
                    - **Λ2** *(float)* -- covariance regularisation
                    - **r** *(float)* -- regularisation coupling strength
            """
            return self._calculate_loss(
                summaries, summary_derivatives, λ, α)

        @jax.jit
        def validation_loss(summaries, derivatives):
            """Jitted loss and auxillary statistics from validation set

            Parameters
            ----------
            summaries : float(n_s, n_summaries)
                The network outputs
            summary_derivatives : float(n_d, n_summaries, n_params)
                The derivative of the network outputs wrt the model parameters

            Returns
            -------
            tuple:
                Fitting statistics calculated on a single validation iteration
                    - **F** *(float(n_params, n_params))* -- Fisher information
                      matrix
                    - **C** *(float(n_summaries, n_summaries))* -- covariance
                      of network outputs
                    - **invC** *(float(n_summaries, n_summaries))* -- inverse
                      covariance of network outputs
                    - **Λ2** *(float)* -- covariance regularisation
                    - **r** *(float)* -- regularisation coupling strength
            """
            F, C, invC, *_ = self._calculate_F_statistics(
                summaries, derivatives)
            _Λ2 = self._get_regularisation(C, invC)
            _r = self._get_regularisation_strength(_Λ2, λ, α)
            return (F, C, invC, _Λ2, _r)

        λ = _check_type(λ, float, "λ")
        ϵ = _check_type(ϵ, float, "ϵ")
        α = self.get_α(λ, ϵ)
        patience = _check_type(patience, int, "patience")
        min_iterations = _check_type(min_iterations, int, "min_iterations")
        max_iterations = _check_type(max_iterations, int, "max_iterations")
        best = _check_boolean(best, "best")
        if self.simulate and (rng is None):
            raise ValueError("`rng` is necessary when simulating.")
        rng = _check_input(rng, (2,), "rng", allow_None=True)
        max_detF, best_w, detF, detC, detinvC, Λ2, r, counter, \
            patience_counter, state, rng = self._set_inputs(
                rng, max_iterations)
        pbar, print_rate, remainder = self._setup_progress_bar(
            print_rate, max_iterations)
        while self._fit_cond(
                (max_detF, best_w, detF, detC, detinvC, Λ2, r, counter,
                 patience_counter, state, rng),
                patience=patience, max_iterations=max_iterations):
            rng, training_key, validation_key, w = get_keys_and_params(
                rng, state)
            summaries, summary_derivatives = self.get_summaries(
                w=w, key=training_key)
            dΛ_dx, results = calculate_loss(summaries, summary_derivatives)
            grad = self.get_gradient(dΛ_dx, w, key=training_key)
            state = self._update(counter, grad, state)
            w = self._get_parameters(state)
            detF, detC, detinvC, Λ2, r = self._update_history(
                results, (detF, detC, detinvC, Λ2, r), counter, 0)
            if self.validate:
                summaries, summary_derivatives = self.get_summaries(
                    w=w, key=training_key, validate=True)
                results = validation_loss(summaries, summary_derivatives)
                detF, detC, detinvC, Λ2, r = self._update_history(
                    results, (detF, detC, detinvC, Λ2, r), counter, 1)
            _detF = np.linalg.det(results[0])
            patience_counter, counter, _, max_detF, __, best_w = \
                jax.lax.cond(
                    np.greater(_detF, max_detF),
                    self._update_loop_vars,
                    lambda inputs: self._check_loop_vars(
                        inputs, min_iterations),
                    (patience_counter, counter, _detF, max_detF, w, best_w))
            self._update_progress_bar(
                pbar, counter, patience_counter, max_detF, detF[counter],
                detC[counter], detinvC[counter], Λ2[counter], r[counter],
                print_rate, max_iterations, remainder)
            counter += 1
        self._update_progress_bar(
            pbar, counter, patience_counter, max_detF, detF[counter - 1],
            detC[counter - 1], detinvC[counter - 1], Λ2[counter - 1],
            r[counter - 1], print_rate, max_iterations, remainder, close=True)
        self.history["max_detF"] = max_detF
        self.best_w = best_w
        self._set_history(
            (detF[:counter],
             detC[:counter],
             detinvC[:counter],
             Λ2[:counter],
             r[:counter]))
        self.state = state
        self.final_w = self._get_parameters(self.state)
        if best:
            w = self.best_w
        else:
            w = self.final_w
        self.set_F_statistics(w, key=rng)

    def _setup_progress_bar(self, print_rate, max_iterations):
        """Construct progress bar

        Parameters
        ----------
        print_rate : int or None
            The rate at which the progress bar is updated (no bar if None)
        max_iterations : int
            The maximum number of iterations, used to setup bar upper limit

        Returns
        -------
        progress bar or None:
            The TQDM progress bar object
        int or None:
            The print rate (after checking for int or None)
        int or None:
            The difference between the max_iterations and the print rate

        Raises
        ------
        TypeError:
            If ``print_rate`` is not an integer
        """
        print_rate = _check_type(
            print_rate, int, "print_rate", allow_None=True)
        if print_rate is not None:
            if max_iterations < 10000:
                pbar = tqdm.tqdm(total=max_iterations)
            else:
                pbar = tqdm.tqdm()
            remainder = max_iterations % print_rate
            return pbar, print_rate, remainder
        else:
            return None, None, None

    def _update_progress_bar(
            self, pbar, counter, patience_counter, max_detF, detF, detC,
            detinvC, Λ2, r, print_rate, max_iterations, remainder,
            close=False):
        """Updates (and closes) progress bar

        Checks whether a pbar is used and is so checks whether the iteration
        coincides with the print rate, or is the last set of iterations within
        the print rate from the last iteration, or if the last iteration has
        been reached and the bar should be closed.

        Parameters
        ----------
        pbar : progress bar object
            The TQDM progress bar
        counter : int
            The value of the current iteration
        patience_counter : int
            The number of iterations where the maximum of the determinant of
            the Fisher information matrix has not increased
        max_detF : float
            Maximum of the determinant of the Fisher information matrix
        detF : float(n_params, n_params)
            Fisher information matrix
        detC : float(n_summaries, n_summaries)
            Covariance of the network summaries
        detinvC : float(n_summaries, n_summaries)
            Inverse covariance of the network summaries
        Λ2 : float
            Value of the regularisation term
        r : float
            Value of the dynamic regularisation coupling strength
        print_rate : int or None
            The number of iterations to run before updating the progress bar
        max_iterations : int
            The maximum number of iterations to run
        remainder : int or None
            The number of iterations before max_iterations to check progress
        close : bool, default=False
            Whether to close the progress bar (on final iteration)
        """
        if print_rate is not None:
            if ((counter % print_rate == 0)
                    and (counter != max_iterations - remainder)) \
                    or (counter == max_iterations - remainder) \
                    or close:
                postfix_dict = dict(
                    patience=patience_counter, detF=detF, detC=detC,
                    detinvC=detinvC, Λ2=Λ2, r=r)
                if (counter != max_iterations - remainder):
                    pbar.update(print_rate)
                elif close:
                    pbar.update(int(counter - pbar.n))
                    postfix_dict["max_detF"] = max_detF
                else:
                    pbar.update(remainder)
                pbar.set_postfix(postfix_dict)
            if close:
                pbar.close()

    def _collect_input(self, key, validate=False):
        """Returns the dataset to be interated over

        Parameters
        ----------
        key: int(2,)
            A random number generator
        validate: bool, default=False
            Whether to use the validation set or not
        """
        raise ValueError("`collect_input` not implemented")

    def _get_batch_summaries(
            self, inputs, w, θ, gradient=False, derivative=False):
        """Vectorised batch calculation of summaries or gradients

        Parameters
        ----------
        inputs: tuple
            - **dΛ_dx** if ``gradient`` *(float(n_per_device, n_summaries) or
              tuple)*

                - **dΛ_dx** *(float(n_per_device, n_summaries))* -- The
                  gradient of the loss function with respect to network outputs
                - **d2Λ_dxdθ** if ``derivative``
                  *(float(n_per_device, n_summaries, n_params))* -- The
                  gradient of the loss function with respect to derivative of
                  network outputs with respect to model parameters
            - **keys** if :func:`~imnn.SimulatorIMNN`
              *(int(n_per_device, 2))* -- The keys for generating simulations
              on-the-fly
            - **d** if :func:`~imnn.NumericalGradientIMNN`
              *(float(n_per_device, input_shape) or tuple)*

                - **d** *(float(n_per_device, input_shape))* -- The simulations
                  to be evaluated
                - **dd_dθ** if ``derivative``
                  *(float(n_per_device, input_shape, n_params))* -- The
                  derivative of the simulations to be evaluated with respect to
                  model parameters
        w: list
            Network model parameters
        θ: float(n_params,)
            The value of the model parameters to generate simulations at/to
            perform the derivative calculation
        gradient: bool
            Whether to do the gradient calculation
        derivative: bool, default=False
            Whether the gradient of loss function with respect to the
            derivative of the network outputs with respect to the model
            parameters is being used

        Returns
        -------
        float(n_devices, n_per_device, n_summaries) or tuple:
            - **x** if not ``gradient`` *(float(n_per_device, n_summaries) or
              tuple)*

                - **x** *(float(n_per_device, n_summaries))* -- The network
                  outputs
                - **dd_dθ** if ``derivative``
                  *(float(n_per_device, n_summaries, n_params))* -- The
                  derivative of the network outputs with respect to model
                  parameters
            - if ``gradient``
                *(tuple)* -- The accumlated and aggregated gradient of the loss
                function with respect to the network parameters
        """
        @jax.vmap
        def fn(inputs):
            return self.get_summary(
                inputs, w, θ, derivative=derivative, gradient=gradient)
        result = fn(inputs)
        if gradient:
            result = self._construct_gradient(result, func="sum")
        return result

    def get_summaries(self, w, key=None, validate=False):
        """Gets all network outputs and derivatives wrt model parameters

        Parameters
        ----------
        w: list or None, default=None
            The network parameters if wanting to calculate the Fisher
            information with a specific set of network parameters
        key: int(2,) or None, default=None
            A random number generator for generating simulations on-the-fly
        validate: bool, default=True
            Whether to get summaries of the validation set

        Returns
        -------
        float(n_s, n_summaries)
            The network outputs
        float(n_d, n_summaries, n_params)
            The derivative of the network outputs wrt the model parameters

        Raises
        ------
        ValueError
            function not implemented in parent class
        """
        raise ValueError(
            "``get_summaries`` not implemented in ``_AggregatedIMNN``")

    def _split_dΛ_dx(self, dΛ_dx):
        """Separates dΛ_dx and d2Λ_dxdθ and reshapes them for aggregation

        Parameters
        ----------
        dΛ_dx: tuple
            - **dΛ_dx** *(float(n_s, n_summaries))* -- The derivative of the
              loss function wrt the network outputs
            - **d2Λ_dxdθ** *(float(n_d, n_summaries, n_params))* -- The
              derivative of the loss function wrt the derivative of the network
              outputs wrt the model parameters

        Raises
        ------
        ValueError
            function not implemented in parent class
        """
        raise ValueError(
            "``_split_dΛ_dx`` not implemented in ``_AggregatedIMNN``")

    def get_gradient(self, dΛ_dx, w, key=None):
        """Aggregates gradients together to update the network parameters

        To avoid having to calculate the gradient with respect to all the
        simulations at once we aggregate by addition the gradient calculation
        by looping over the simulations again and combining them with the
        derivative of the loss function with respect to the network outputs
        (and their derivatives with respect to the model parameters). Whilst
        this is expensive, it is necessary since we cannot make a stochastic
        estimate of the Fisher information accurately and therefore we need to
        use all the simulations available - which is probably too large to fit
        in memory.

        Parameters
        ----------
        dΛ_dx : tuple
            - **dΛ_dx** *float(n_s, n_params, n_summaries)* -- the derivative
              of the loss function with respect to network summaries
            - **d2Λ_dxdθ** *float(n_d, n_summaries, n_params)* -- the
              derivative of the loss function with respect to the derivative of
              network summaries with respect to model parameters
        w : list
            Network parameters
        key : None or int(2,)
            Random number generator used in SimulatorIMNN

        Returns
        -------
        list:
            The gradient of the loss function with respect to the network
            parameters calculated by aggregating

        Raises
        ------
        ValueError
            function not implemented in parent class
        """
        raise ValueError(
            "``get_gradient`` not implemented in ``_AggregatedIMNN``")

    def _construct_gradient(self, layers, aux=None, func="zeros"):
        """Multiuse function to iterate over tuple of network parameters

        The options are:
            - ``"zeros"`` -- to create an empty gradient array
            - ``"einsum"`` -- to combine tuple of ``dx_dw`` with ``dΛ_dx``
            - ``"derivative_einsum"`` -- to combine tuple of ``d2x_dwdθ`` with
              ``d2Λ_dxdθ``
            - ``"sum"`` -- to reduce sum batches of gradients on the first axis

        Parameters
        ----------
        layers: tuple
            The tuple of tuples of arrays to be iterated over
        aux: float(various shapes)
            parameter to pass dΛ_dx and d2Λ_dxdθ to einsum
        func: str
            Option for the function to apply
                - ``"zeros"`` -- to create an empty gradient array
                - ``"einsum"`` -- to combine tuple of ``dx_dw`` with ``dΛ_dx``
                - ``"derivative_einsum"`` -- to combine tuple of ``d2x_dwdθ``
                  with ``d2Λ_dxdθ``
                - ``"sum"`` -- to reduce sum batches of gradients on the first
                  axis

        Returns
        -------
        tuple:
            Tuple of objects like the gradient of the loss function with
            respect to the network parameters

        Raises
        ------
        ValueError
            If applied function is not implemented
        """
        if func == "zeros":
            def func(x):
                return np.zeros_like(x)
        elif func == "einsum":
            def func(x):
                return np.einsum("i,i...->...", aux, x)
        elif func == "derivative_einsum":
            def func(x):
                return np.einsum("ij,ij...->...", aux, x)
        elif func == "sum":
            def func(x):
                return np.sum(x, 0)
        else:
            raise ValueError("`func` must be `'zeros'`, `'einsum'`, " +
                             "`'derivative_einsum'` or `'sum'`")

        value_flat, value_tree = tree_flatten(layers)
        transformed_flat = list(map(func, value_flat))
        return tree_unflatten(value_tree, transformed_flat)
