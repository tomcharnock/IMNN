import jax
import jax.numpy as np
from functools import partial
from imnn.imnn._aggregated_imnn import _AggregatedIMNN
from imnn.imnn import SimulatorIMNN
from imnn.utils import jacrev, value_and_jacrev, add_nested_pytrees
from imnn.utils.utils import _check_splitting


class AggregatedSimulatorIMNN(_AggregatedIMNN, SimulatorIMNN):
    """Information maximising neural network fit with simulations on-the-fly

    Defines the function to get simulations and compress them using an XLA
    compilable simulator.

    The outline of the fitting procedure is that a set of :math:`i\\in[1, n_s]`
    random number generators are generated and used to generate a set of
    :math:`n_s` simulations,
    :math:`{\\bf d}^i={\\rm simulator}({\\rm seed}^i, \\theta^\\rm{fid})` at
    the fiducial model parameters, :math:`\\theta^\\rm{fid}`, and these are
    passed direrectly through a network :math:`f_{{\\bf w}}({\\bf d})` with
    network parameters :math:`{\\bf w}` to obtain network outputs
    :math:`{\\bf x}^i` and autodifferentiation is used to get the derivative of
    :math:`n_d` of these outputs with respect to the physical model parameters,
    :math:`\\partial{{\\bf x}^i}/\\partial\\theta_\\alpha`, where
    :math:`\\alpha` labels the physical parameter. With :math:`{\\bf x}^i` and
    :math:`\\partial{{\\bf x}^i}/\\partial\\theta_\\alpha` the covariance

    .. math::
        C_{ab} = \\frac{1}{n_s-1}\\sum_{i=1}^{n_s}(x^i_a-\\mu^i_a)
        (x^i_b-\\mu^i_b)

    and the derivative of the mean of the network outputs with respect to the
    model parameters

    .. math::
        \\frac{\\partial\\mu_a}{\\partial\\theta_\\alpha} = \\frac{1}{n_d}
        \\sum_{i=1}^{n_d}\\frac{\\partial{x^i_a}}{\\partial\\theta_\\alpha}

    can be calculated and used form the Fisher information matrix

    .. math::
        F_{\\alpha\\beta} = \\frac{\\partial\\mu_a}{\\partial\\theta_\\alpha}
        C^{-1}_{ab}\\frac{\\partial\\mu_b}{\\partial\\theta_\\beta}.

    The loss function is then defined as

    .. math::
        \\Lambda = -\\log|{\\bf F}| + r(\\Lambda_2) \\Lambda_2

    Since any linear rescaling of a sufficient statistic is also a sufficient
    statistic the negative logarithm of the determinant of the Fisher
    information matrix needs to be regularised to fix the scale of the network
    outputs. We choose to fix this scale by constraining the covariance of
    network outputs as

    .. math::
        \\Lambda_2 = ||{\\bf C}-{\\bf I}|| + ||{\\bf C}^{-1}-{\\bf I}||

    Choosing this constraint is that it forces the covariance to be
    approximately parameter independent which justifies choosing the covariance
    independent Gaussian Fisher information as above. To avoid having a dual
    optimisation objective, we use a smooth and dynamic regularisation strength
    which turns off the regularisation to focus on maximising the Fisher
    information when the covariance has set the scale

    .. math::
        r(\\Lambda_2) = \\frac{\\lambda\\Lambda_2}{\\Lambda_2-\\exp
        (-\\alpha\\Lambda_2)}.

    To enable the use of large data (or networks) the whole procedure is
    aggregated. This means that the generation and passing of the simulations
    through the network is farmed out to the desired XLA devices, and
    recollected, ``n_per_device`` inputs at a time. These are then used to
    calculate the automatic gradient of the loss function with respect to the
    calculated summaries and derivatives,
    :math:`\\partial\\Lambda/\\partial{\\bf x}^i` (which is a fairly small
    computation as long as ``n_summaries`` and ``n_s`` {and ``n_d``} are not
    huge). Once this is calculated, the simulations are passed through the
    network AGAIN this time calculating the Jacobian of the network output with
    respect to the network parameters
    :math:`\\partial{\\bf x}^i/\\partial{\\bf w}` which is then combined via
    the chain rule to get

    .. math::
        \\frac{\\partial\\Lambda}{\\partial{\\bf w}} =
        \\frac{\\partial\\Lambda}{\\partial{\\bf x}^i}
        \\frac{\\partial{\\bf x}^i}{\\partial{\\bf w}}

    This can then be passed to the optimiser.

    Parameters
    ----------
    n_remaining: int
        The number simulations where only the fiducial simulations are
        calculated. This is zero if ``n_s`` is equal to ``n_d``.
    n_iterations : int
        Number of iterations through the main summarising loop
    n_remaining_iterations : int
        Number of iterations through the remaining simulations used for quick
        loops with no derivatives
    batch_shape: tuple
        The shape which ``n_d`` should be reshaped to for aggregating.
        ``n_d // (n_devices * n_per_device), n_devices, n_per_device``
    remaining_batch_shape: tuple
        The shape which ``n_s - n_d`` should be reshaped to for aggregating.
        ``(n_s - n_d) // (n_devices * n_per_device), n_devices, n_per_device``

    Methods
    -------
    simulator:
        A function for generating a simulation on-the-fly (XLA compilable)

    """
    def __init__(self, n_s, n_d, n_params, n_summaries, input_shape, θ_fid,
                 model, optimiser, key_or_state, simulator, host, devices,
                 n_per_device):
        """Constructor method

        Initialises all IMNN attributes, constructs neural network and its
        initial parameter values and creates history dictionary. Also fills the
        simulation attributes (and validation if available).

        Parameters
        ----------
        n_s : int
            Number of simulations used to calculate summary covariance
        n_d : int
            Number of simulations used to calculate mean of summary derivative
        n_params : int
            Number of model parameters
        n_summaries : int
            Number of summaries, i.e. outputs of the network
        input_shape : tuple
            The shape of a single input to the network
        θ_fid : float(n_params,)
            The value of the fiducial parameter values used to generate inputs
        model : tuple, len=2
            Tuple containing functions to initialise neural network
            ``fn(rng: int(2), input_shape: tuple) -> tuple, list`` and the
            neural network as a function of network parameters and inputs
            ``fn(w: list, d: float(None, input_shape)) -> float(None, n_summari
            es)``.
            (Essentibly stax-like, see `jax.experimental.stax <https://jax.read
            thedocs.io/en/stable/jax.experimental.stax.html>`_))
        optimiser : tuple, len=3
            Tuple containing functions to generate the optimiser state
            ``fn(x0: list) -> :obj:state``, to update the state from a list of
            gradients ``fn(i: int, g: list, state: :obj:state) -> :obj:state``
            and to extract network parameters from the state
            ``fn(state: :obj:state) -> list``.
            (See `jax.experimental.optimizers <https://jax.readthedocs.io/en/st
            able/jax.experimental.optimizers.html>`_)
        key_or_state : int(2) or :obj:state
            Either a stateless random number generator or the state object of
            an preinitialised optimiser
        simulator : fn
            A function that generates a single simulation from a random number
            generator and a tuple (or array) of parameter values at which to
            generate the simulations. For the purposes of use in LFI/ABC
            afterwards it is also useful for the simulator to be able to
            broadcast to a batch of simulations on the zeroth axis
            ``fn(int(2,), float([None], n_params)) ->
            float([None], input_shape)``
        host: jax.device
            The main device where the Fisher calculation is performed
        devices: list
            A list of the available jax devices (from ``jax.devices()``)
        n_per_device: int
            Number of simulations to handle at once, this should be as large as
            possible without letting the memory overflow for the best
            performance
        """
        SimulatorIMNN.__init__(
            self=self,
            n_s=n_s,
            n_d=n_d,
            n_params=n_params,
            n_summaries=n_summaries,
            input_shape=input_shape,
            θ_fid=θ_fid,
            model=model,
            key_or_state=key_or_state,
            optimiser=optimiser,
            simulator=simulator)
        _AggregatedIMNN.__init__(
            self=self,
            host=host,
            devices=devices,
            n_per_device=n_per_device)

    def _set_shapes(self):
        """Calculates the shapes for batching over different devices

        Raises
        ------
        ValueError
            If the difference between n_s and n_d won't scale over xla devices
        """
        self.n_remaining = self.n_s - self.n_d
        _check_splitting(self.n_remaining, "n_s - n_d", self.n_devices,
                         self.n_per_device)
        self.n_iterations = self.n_d // (self.n_devices * self.n_per_device)
        self.n_remaining_iterations = self.n_remaining \
            // (self.n_devices * self.n_per_device)
        self.batch_shape = (
            self.n_devices,
            self.n_iterations,
            self.n_per_device)
        self.remaining_batch_shape = (
            self.n_devices,
            self.n_remaining_iterations,
            self.n_per_device)

    def get_summary(self, input, w, θ, derivative=False, gradient=False):
        """ Returns a single summary of a simulation or its gradient

        Parameters
        ----------
        input : float(input_shape) or tuple
            A random number generator for a single simulation to pass through
            the network or a tuple of either (if ``gradient`` and
            ``not derivative``)

                - **dΛ_dx** *float(input_shape, n_params)* -- the derivative of
                  the loss function with respect to a network summary
                - **key** *int(2,)* -- A random number generator for a single
                  simulation

            or (if ``gradient`` and ``derivative``)

                - tuple (gradients)
                    - **dΛ_dx** *float(input_shape, n_params)* -- the
                      derivative of the loss function with respect to a network
                      summary
                    - **d2Λ_dxdθ** *float(input_shape, n_params)* -- the
                      derivative of the loss function with respect to the
                      derivative of a network summary with respect to model
                      parameters

                - **key** *int(2,)* -- A random number generator for a single
                  simulation
        w : list
            The network parameters
        θ : float(n_params,)
            The value of the parameters to generate the simulation at
            (fiducial), unused if not simulating on the fly
        derivative : bool, default=False
            Whether a derivative of the simulation with respect to model
            parameters is also passed.
        gradient : bool, default=False
            Whether to calculate the gradient with respect to model parameters

        Returns
        -------
        tuple (if ``gradient``):
            The gradient of the loss function with respect to model parameters
        float(n_summaries,) (if ``not gradient``):
            The output of the network
        float(n_summaries, n_params) (if ``derivative`` and ``not gradient``):
            The derivative of the output of the network wrt model parameters
        """
        def fn(key, w, θ):
            """ Returns a compressed simulation from the network

            Parameters
            ----------
            key : int(2,)
                The random number generator for the simulation
            w : list
                The network parameters
            θ : float(n_params,)
                The model parameters to generate the simulation at

            Returns
            ------
            float(n_summaries,):
                The summary from the output of the network
            """
            return self.model(w, self.simulator(key, θ))

        def grad_fn(key, w, θ):
            """ Derivative of the network outputs wrt to inputs and parameters

            Parameters
            ----------
            key : int(2,)
                The random number generator for the simulation
            w : list
                The network parameters
            θ : float(n_params,)
                The model parameters to generate the simulation at

            Returns
            ------
            float(n_summaries, n_params):
                The derivative of the network output with respect to the model
                parameters
            list:
                The derivative of the network outputs with respect to the
                network parameters
            """
            dx_dw, dx_dθ = jacrev(fn, argnums=(1, 2))(key, w, θ)
            return dx_dθ, dx_dw

        if derivative and gradient:
            dΛ_dx, key = input
            dΛ_dx, d2Λ_dxdθ = dΛ_dx
            d2x_dwdθ, dx_dw = jacrev(
                grad_fn, argnums=1, has_aux=True)(key, w, θ)
            return add_nested_pytrees(
                self._construct_gradient(dx_dw, aux=dΛ_dx, func="einsum"),
                self._construct_gradient(
                    d2x_dwdθ, aux=d2Λ_dxdθ, func="derivative_einsum"))
        elif derivative:
            return value_and_jacrev(fn, argnums=2)(input, w, θ)
        elif gradient:
            dΛ_dx, key = input
            dx_dw = jacrev(fn, argnums=1)(key, w, θ)
            return self._construct_gradient(dx_dw, aux=dΛ_dx, func="einsum")
        else:
            return fn(input, w, θ)

    def _collect_input(self, key, validate=False):
        """Returns the keys for generating simulations on-the-fly

        Parameters
        ----------
        key : None or int(2,)
            Random number generators for generating simulations
        validate : bool, default=False
            Whether to return the set for validation or for fitting
            (always False)
        """
        keys = np.array(jax.random.split(key, num=self.n_s))
        main = [_key for _key in keys[:self.n_d].reshape(
            self.batch_shape + (2,))]
        remaining = [_key for _key in keys[self.n_d:].reshape(
            self.remaining_batch_shape + (2,))]
        return main, remaining

    def get_summaries(self, w, key, validate=False):
        """Gets all network outputs and derivatives wrt model parameters

        Loops through the generated random number generators on each XLA device
        to generate a simulation and then pass them through the network to get
        the network outputs. These are then pushed back to the host for the
        computation of the loss function.

        The fiducial simulations which have a derivative with respect to the
        model parameters counterpart are processed first and then the remaining
        fiducial simulations are processed and concatenated.

        Parameters
        ----------
        w : list or None, default=None
            The network parameters if wanting to calculate the Fisher
            information with a specific set of network parameters
        key : int(2,)
            A random number generator for generating simulations on-the-fly
        validate : bool, default=False
            Whether to get summaries of the validation set (always False)

        Returns
        -------
        float(n_s, n_summaries):
            The set of all network outputs used to calculate the covariance
        float(n_d, n_summaries, n_params):
            The set of all the derivatives of the network outputs with respect
            to model parameters
        """
        main, remaining = self._collect_input(key, validate=validate)

        x = [np.zeros((self.n_s // (self.n_per_device * self.n_devices),
                       self.n_per_device,
                       self.n_summaries))
             for i in range(self.n_devices)]
        dx_dθ = [np.zeros((self.n_iterations,
                          self.n_per_device,
                          self.n_summaries,
                          self.n_params))
                 for i in range(self.n_devices)]

        for i in range(self.n_iterations):
            for j, (fn, dataset) in enumerate(zip(
                    self.batch_summaries_with_derivatives,
                    main)):
                _x, _dx_dθ = fn(dataset[i], w)
                x[j] = jax.ops.index_update(x[j], jax.ops.index[i], _x)
                dx_dθ[j] = jax.ops.index_update(
                    dx_dθ[j], jax.ops.index[i], _dx_dθ)

        for i in range(self.n_remaining_iterations):
            for j, (fn, dataset) in enumerate(zip(
                    self.batch_summaries, remaining)):
                x[j] = jax.ops.index_update(
                    x[j],
                    jax.ops.index[i + self.n_iterations],
                    fn(dataset[i], w))

        if self.n_devices > 1:
            x = [jax.device_put(_x, device=self.host)
                 for _x in x]
            dx_dθ = [jax.device_put(_dx_dθ, device=self.host)
                     for _dx_dθ in dx_dθ]

        x = np.stack(x, 0).reshape((self.n_s, self.n_summaries))
        dx_dθ = np.stack(dx_dθ, 0).reshape(
            (self.n_d, self.n_summaries, self.n_params))
        return x, dx_dθ

    def _split_dΛ_dx(self, dΛ_dx):
        """ Returns the gradient of loss function wrt summaries (derivatives)

        The gradient of loss function with respect to network outputs and
        their derivatives with respect to model parameters has to be reshaped
        and aggregated onto each XLA device matching the format keys are
        generated for generating simulations.

        Parameters
        ----------
        dΛ_dx : tuple
            - **dΛ_dx** *float(n_s, n_params, n_summaries)* -- the derivative
              of the loss function with respect to network summaries
            - **d2Λ_dxdθ** *float(n_d, n_summaries, n_params)* -- the
              derivative of the loss function with respect to the derivative of
              network summaries with respect to model parameters

        Return
        -----

        """
        main_gradients = [
            jax.device_put(val, device=device) for val, device in zip(
                dΛ_dx[0][:self.n_d].reshape(
                    self.batch_shape + (self.n_summaries,)),
                self.devices)]
        main_derivative_gradients = [
            jax.device_put(val, device=device) for val, device in zip(
                dΛ_dx[1].reshape(
                    self.batch_shape + (self.n_summaries, self.n_params)),
                self.devices)]
        remaining_gradients = [
            jax.device_put(val, device=device) for val, device in zip(
                dΛ_dx[0][self.n_d:].reshape(
                    self.remaining_batch_shape + (self.n_summaries,)),
                self.devices)]
        return main_gradients, main_derivative_gradients, remaining_gradients

    def get_gradient(self, dΛ_dx, w, key=None):
        """Aggregates gradients together to update the network parameters

        To avoid having to calculate the gradient with respect to all the
        simulations at once we aggregate by addition the gradient calculation
        by looping over the keys for generating the simulations again which we
        then combine with the derivative of the loss function with respect to
        the network outputs (and their derivatives with respect to the model
        parameters). Whilst this is expensive, it is necessary since we cannot
        make a stochastic estimate of the Fisher information accurately and
        therefore we need to use all the simulations available - which is
        probably too large to fit in memory.

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
            Random number generator

        Returns
        -------
        list:
            The gradient of the loss function with respect to the network
            parameters calculated by aggregating
        """
        main_gradients, main_derivative_gradients, remaining_gradients = \
            self._split_dΛ_dx(dΛ_dx)
        main, remaining = self._collect_input(key, validate=False)

        gradient = [
            jax.jit(
                partial(self._construct_gradient, func="zeros"),
                device=device)(w)
            for device in self.devices]

        for i in range(self.n_iterations):
            for j, (fn, dataset, grad, der_grad, device) in enumerate(
                    zip(self.batch_gradients_with_derivatives, main,
                        main_gradients, main_derivative_gradients,
                        self.devices)):
                gradient[j] = jax.jit(add_nested_pytrees, device=device)(
                    gradient[j],
                    fn(((grad[i], der_grad[i]), dataset[i]), w))
        for i in range(self.n_remaining_iterations):
            for j, (fn, dataset, grad, device) in enumerate(
                    zip(self.batch_gradients, remaining, remaining_gradients,
                        self.devices)):
                gradient[j] = jax.jit(add_nested_pytrees, device=device)(
                    gradient[j],
                    fn((grad[i], dataset[i]), w))

        return list(jax.jit(add_nested_pytrees, device=self.host)(*gradient))
