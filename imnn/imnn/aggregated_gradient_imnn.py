import jax
import jax.numpy as np
import tensorflow as tf
from functools import partial
from imnn.imnn._aggregated_imnn import _AggregatedIMNN
from imnn.imnn import GradientIMNN
from imnn.utils import value_and_jacrev, jacrev, add_nested_pytrees
from imnn.utils.utils import _check_splitting, _check_boolean, _check_type


class AggregatedGradientIMNN(_AggregatedIMNN, GradientIMNN):
    """Information maximising neural network fit using known derivatives


    The outline of the fitting procedure is that a set of :math:`i\\in[1, n_s]`
    simulations :math:`{\\bf d}^i` originally generated at fiducial model
    parameter :math:`{\\bf\\theta}^\\rm{fid}`, and their derivatives
    :math:`\\partial{\\bf d}^i/\\partial\\theta_\\alpha` with respect to
    model parameters are used. The fiducial simulations, :math:`{\\bf d}^i`,
    are passed through a network to obtain summaries, :math:`{\\bf x}^i`, and
    the jax automatic derivative of these summaries with respect to the inputs
    are calculated :math:`\\partial{\\bf x}^i\\partial{\\bf d}^j\\delta_{ij}`.
    The chain rule is then used to calculate

    .. math::
        \\frac{\\partial{\\bf x}^i}{\\partial\\theta_\\alpha} =
        \\frac{\\partial{\\bf x}^i}{\\partial{\\bf d}^j}
        \\frac{\\partial{\\bf d}^j}{\\partial\\theta_\\alpha}

    With :math:`{\\bf x}^i` and
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
    aggregated. This means that the passing of the simulations through the
    network is farmed out to the desired XLA devices, and recollected,
    ``n_per_device`` inputs at a time. These are then used to calculate the
    automatic gradient of the loss function with respect to the calculated
    summaries and derivatives, :math:`\\partial\\Lambda/\\partial{\\bf x}^i`
    (which is a fairly small computation as long as ``n_summaries`` and ``n_s``
    {and ``n_d``} are not huge). Once this is calculated, the simulations are
    passed through the network AGAIN this time calculating the Jacobian of the
    network output with respect to the network parameters
    :math:`\\partial{\\bf x}^i/\\partial{\\bf w}` which is then combined via
    the chain rule to get

    .. math::
        \\frac{\\partial\\Lambda}{\\partial{\\bf w}} =
        \\frac{\\partial\\Lambda}{\\partial{\\bf x}^i}
        \\frac{\\partial{\\bf x}^i}{\\partial{\\bf w}}

    This can then be passed to the optimiser.

    Parameters
    ----------
    fiducial : float(n_s, input_shape)
        The simulations generated at the fiducial model parameter values used
        for calculating the covariance of network outputs (for fitting)
    derivative : float(n_d, input_shape, n_params)
        The derivative of the simulations with respect to the model parameters
        (for fitting)
    validation_fiducial : float(n_s, input_shape) or None
        The simulations generated at the fiducial model parameter values used
        for calculating the covariance of network outputs (for validation)
    validation_derivative : float(n_d, input_shape, n_params) or None
        The derivative of the simulations with respect to the model parameters
        (for validation)
    main : list of tf.data.Dataset().as_numpy_iterators()
        The simulations generated at the fiducial model parameter values used
        for calculating the covariance of network outputs and their
        derivatives with respect to the physical model parameters
        (for fitting). These are served ``n_per_device`` at a time as a
        numpy iterator from a TensorFlow dataset.
    remaining : list of tf.data.Dataset().as_numpy_iterators()
        The ``n_s - n_d`` simulations generated at the fiducial model parameter
        values used for calculating the covariance of network outputs with a
        derivative counterpart (for fitting). These are served ``n_per_device``
        at a time as a numpy iterator from a TensorFlow dataset.
    validation_main : list of tf.data.Dataset().as_numpy_iterators()
        The simulations generated at the fiducial model parameter values used
        for calculating the covariance of network outputs and their
        derivatives with respect to the physical model parameters
        (for validation). These are served ``n_per_device`` at a time as a
        numpy iterator from a TensorFlow dataset.
    validation_remaining : list of tf.data.Dataset().as_numpy_iterators()
        The ``n_s - n_d`` simulations generated at the fiducial model parameter
        values used for calculating the covariance of network outputs with a
        derivative counterpart (for validation). Served ``n_per_device``
        at time as a numpy iterator from a TensorFlow dataset.
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
    """
    def __init__(self, n_s, n_d, n_params, n_summaries, input_shape, θ_fid,
                 model, optimiser, key_or_state, fiducial, derivative, host,
                 devices, n_per_device, validation_fiducial=None,
                 validation_derivative=None, prefetch=None, cache=False):
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
        fiducial : float(n_s, input_shape)
            The simulations generated at the fiducial model parameter values
            used for calculating the covariance of network outputs
            (for fitting)
        derivative : float(n_d, input_shape, n_params)
            The derivative of the simulations with respect to the
            model parameters (for fitting)
        host: jax.device
            The main device where the Fisher calculation is performed
        devices: list
            A list of the available jax devices (from ``jax.devices()``)
        n_per_device: int
            Number of simulations to handle at once, this should be as large as
            possible without letting the memory overflow for the best
            performance
        validation_fiducial : float(n_s, input_shape) or None, default=None
            The simulations generated at the fiducial model parameter values
            used for calculating the covariance of network outputs
            (for validation)
        validation_derivative : float(n_d, input_shape, n_params) or None
            The derivative of the simulations with respect to the
            model parameters (for validation)
        prefetch : tf.data.AUTOTUNE or int or None, default=None
            How many simulation to prefetch in the tensorflow dataset
        cache : bool, default=False
            Whether to cache simulations in the tensorflow datasets
        """
        GradientIMNN.__init__(
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
            fiducial=fiducial,
            derivative=derivative,
            validation_fiducial=validation_fiducial,
            validation_derivative=validation_derivative)
        _AggregatedIMNN.__init__(
            self=self,
            host=host,
            devices=devices,
            n_per_device=n_per_device)
        self._set_dataset(prefetch, cache)

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

    def _set_dataset(self, prefetch, cache):
        """ Collates data into loopable tensorflow dataset iterations

        Parameters
        ----------
        prefetch : tf.data.AUTOTUNE or int or None
            How many simulation to prefetch in the tensorflow dataset
        cache : bool
            Whether to cache simulations in the tensorflow datasets

        Raises
        ------
        ValueError
            If cache is None
        TypeError
            If cache is wrong type
        """
        cache = _check_boolean(cache, "cache")
        prefetch = _check_type(prefetch, int, "prefetch", allow_None=True)
        self.remaining = [
            tf.data.Dataset.from_tensor_slices(
                fiducial)
            for fiducial in self.fiducial[self.n_d:].reshape(
                self.remaining_batch_shape + self.input_shape)]

        self.main = [
            tf.data.Dataset.from_tensor_slices(
                (fiducial, derivative))
            for fiducial, derivative in zip(
                self.fiducial[:self.n_d].reshape(
                    self.batch_shape + self.input_shape),
                self.derivative.reshape(
                    self.batch_shape + self.input_shape + (self.n_params,)))]

        if cache:
            self.remaining = [
                remaining.cache()
                for remaining in self.remaining]
            self.main = [
                main.cache()
                for main in self.main]
        if prefetch is not None:
            self.remaining = [
                remaining.prefetch(prefetch)
                for remaining in self.remaining]
            self.main = [
                main.prefetch(prefetch)
                for main in self.main]

        self.main = [
            main.repeat().as_numpy_iterator()
            for main in self.main]
        self.remaining = [
            remaining.repeat().as_numpy_iterator()
            for remaining in self.remaining]

        if self.validate:
            self.validation_remaining = [
                tf.data.Dataset.from_tensor_slices(
                    fiducial)
                for fiducial in self.validation_fiducial[self.n_d:].reshape(
                    self.remaining_batch_shape + self.input_shape)]

            self.validation_main = [
                tf.data.Dataset.from_tensor_slices(
                    (fiducial, derivative))
                for fiducial, derivative in zip(
                    self.validation_fiducial[:self.n_d].reshape(
                        self.batch_shape + self.input_shape),
                    self.validation_derivative.reshape(
                        self.batch_shape + self.input_shape +
                        (self.n_params,)))]

            if cache:
                self.validation_remaining = [
                    remaining.cache()
                    for remaining in self.validation_remaining]
                self.validation_main = [
                    main.cache()
                    for main in self.validation_main]
            if prefetch is not None:
                self.validation_remaining = [
                    remaining.prefetch(prefetch)
                    for remaining in self.validation_remaining]
                self.validation_main = [
                    main.prefetch(prefetch)
                    for main in self.validation_main]

            self.validation_main = [
                main.repeat().as_numpy_iterator()
                for main in self.validation_main]
            self.validation_remaining = [
                remaining.repeat().as_numpy_iterator()
                for remaining in self.validation_remaining]

    def get_summary(self, input, w, θ, derivative=False, gradient=False):
        """ Returns a single summary of a simulation or its gradient

        Parameters
        ----------
        input : float(input_shape) or tuple
            A single simulation to pass through the network or a tuple of
            either (if ``gradient`` and ``not derivative``)

                - **dΛ_dx** *float(input_shape, n_params)* -- the derivative of
                  the loss function with respect to a network summary
                - **d** *float(input_shape)* -- a simulation to compress with
                  the network

            or (if ``gradient`` and ``derivative``)

                - tuple (gradients)

                    - **dΛ_dx** *float(input_shape, n_params)* -- the
                      derivative of the loss function with respect to a network
                      summary
                    - **d2Λ_dxdθ** *float(input_shape, n_params)* -- the
                      derivative of the loss function with respect to the
                      derivative of a network summary with respect to model
                      parameters

                - tuple (simulations)

                    - **d** *float(input_shape)* -- a simulation to compress
                      with the network
                    - **dd_dθ** *float(input_shape, n_params)* -- the
                      derivative of a simulation with respect to model
                      parameters

            or (if ``derivative`` and ``not gradient``)

                - **d** *float(input_shape)* -- a simulation to compress with
                  the network
                - **dd_dθ** *float(input_shape, n_params)* -- the derivative of
                  a simulation with respect to model parameters
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
        def fn(d, w):
            """ Returns a compressed simulation from the network

            Parameters
            ----------
            d : float(input_shape)
                The simulation to be compressed
            w : list
                The network parameters

            Returns
            ------
            float(n_summaries,):
                The summary from the output of the network
            """
            return self.model(w, d)

        def grad_fn(d, w, dd_dθ):
            """ Derivative of the network outputs wrt to inputs and parameters

            The derivative of the network with respect to the input data is
            combined via the chain rule with the derivative of the simulation
            with respect to the model parameters to obtain the derivative of
            the network outputs with respect to the model parameters, i.e.

            .. math::
                \frac{\\partial{{\\bf x}^i}}{\\partial\\theta_\\alpha} =
                \\frac{\\partial{\\bf x}^i}{\\partial{\\bf d}^j}
                \\frac{\\partial{\\bf d}^j}{\\partial\\theta_\\alpha}

            Parameters
            ----------
            d : float(input_shape)
                The simulation to be compressed
            w : list
                The network parameters
            dd_dθ : float(input_shape, n_params)
                The derivative of the simulation wrt model parameters

            Returns
            ------
            float(n_summaries, n_params):
                The derivative of the network output with respect to the model
                parameters
            list:
                The derivative of the network outputs with respect to the
                network parameters
            """
            dx_dd, dx_dw = jacrev(fn, argnums=(0, 1))(d, w)
            return np.einsum("i...,...j->ij", dx_dd, dd_dθ), dx_dw

        if derivative and gradient:
            (dΛ_dx, d2Λ_dxdθ), (d, dd_dθ) = input
            d2x_dwdθ, dx_dw = jacrev(
                grad_fn, argnums=1, has_aux=True)(d, w, dd_dθ)
            return add_nested_pytrees(
                self._construct_gradient(dx_dw, aux=dΛ_dx, func="einsum"),
                self._construct_gradient(
                    d2x_dwdθ, aux=d2Λ_dxdθ, func="derivative_einsum"))
        elif derivative:
            d, dd_dθ = input
            x, dx_dd = value_and_jacrev(fn, argnums=0)(d, w)
            return x, np.einsum("i...,...j->ij", dx_dd, dd_dθ)
        elif gradient:
            dΛ_dx, d = input
            dx_dw = jacrev(fn, argnums=1)(d, w)
            return self._construct_gradient(dx_dw, aux=dΛ_dx, func="einsum")
        else:
            return fn(input, w)

    def _collect_input(self, key, validate=False):
        """ Returns validation or fitting sets

        Parameters
        ----------
        key : None or int(2,)
            Random number generators not used in this case
        validate : bool
            Whether to return the set for validation or for fitting

        Returns
        -------
        list of tf.data.Dataset().as_numpy_iterators:
            The iterators for the main loop including simulations and their
            derivatives for fitting or validation
        list of tf.data.Dataset().as_numpy_iterators:
            The iterators for the remaining loop simulations for fitting or
            validation
        """
        if validate:
            remaining = self.validation_remaining
            main = self.validation_main
        else:
            remaining = self.remaining
            main = self.main
        return main, remaining

    def get_summaries(self, w, key=None, validate=False):
        """Gets all network outputs and derivatives wrt model parameters

        Selects either the fitting or validation sets and loops through the
        iterator on each XLA device to pass them through the network to get the
        network outputs. These are then pushed back to the host for the
        computation of the loss function.

        The fiducial simulations which have a derivative with respect to the
        model parameters counterpart are processed first and then the remaining
        fiducial simulations are processed and concatenated.

        Parameters
        ----------
        w : list or None, default=None
            The network parameters if wanting to calculate the Fisher
            information with a specific set of network parameters
        key : int(2,) or None, default=None
            A random number generator for generating simulations on-the-fly
        validate : bool, default=False
            Whether to get summaries of the validation set

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
                    self.batch_summaries_with_derivatives, main)):
                _x, _dx_dθ = fn(next(dataset), w)
                x[j] = jax.ops.index_update(x[j], jax.ops.index[i], _x)
                dx_dθ[j] = jax.ops.index_update(
                    dx_dθ[j], jax.ops.index[i], _dx_dθ)

        for i in range(self.n_remaining_iterations):
            for j, (fn, dataset) in enumerate(zip(
                    self.batch_summaries, remaining)):
                x[j] = jax.ops.index_update(
                    x[j],
                    jax.ops.index[i + self.n_iterations],
                    fn(next(dataset), w))

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
        and aggregated onto each XLA device matching the format that the
        tensorflow dataset feeds simulations.

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
                    fn(((grad[i], der_grad[i]), next(dataset)), w))

        for i in range(self.n_remaining_iterations):
            for j, (fn, dataset, grad, device) in enumerate(
                    zip(self.batch_gradients, remaining, remaining_gradients,
                        self.devices)):
                gradient[j] = jax.jit(add_nested_pytrees, device=device)(
                    gradient[j],
                    fn((grad[i], next(dataset)), w))

        return list(jax.jit(add_nested_pytrees, device=self.host)(*gradient))
