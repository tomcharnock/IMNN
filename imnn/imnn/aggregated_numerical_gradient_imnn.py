import jax
import jax.numpy as np
import tensorflow as tf
from functools import partial
from imnn.imnn._aggregated_imnn import _AggregatedIMNN
from imnn.imnn import NumericalGradientIMNN
from imnn.utils import jacrev, add_nested_pytrees
from imnn.utils.utils import _check_boolean, _check_type


class AggregatedNumericalGradientIMNN(_AggregatedIMNN, NumericalGradientIMNN):
    """Information maximising neural network fit using numerical derivatives

    The outline of the fitting procedure is that a set of :math:`i\\in[1, n_s]`
    simulations :math:`{\\bf d}^i` originally generated at fiducial model
    parameter :math:`{\\bf\\theta}^\\rm{fid}`, and a set of
    :math:`i\\in[1, n_d]` simulations,
    :math:`\\{{\\bf d}_{\\alpha^-}^i, {\\bf d}_{\\alpha^+}^i\\}`, generated
    with the same seed at each :math:`i` generated at
    :math:`{\\bf\\theta}^\\rm{fid}` apart from at parameter label
    :math:`\\alpha` with values

    .. math::
        \\theta_{\\alpha^-} = \\theta_\\alpha^\\rm{fid}-\\delta\\theta_\\alpha

    and

    .. math::
        \\theta_{\\alpha^+} = \\theta_\\alpha^\\rm{fid}+\\delta\\theta_\\alpha

    where :math:`\\delta\\theta_\\alpha` is a :math:`n_{params}` length vector
    with the :math:`\\alpha` element having a value which perturbs the
    parameter :math:`\\theta^{\\rm fid}_\\alpha`. This means there are
    :math:`2\\times n_{params}\\times n_d` simulations used to calculate the
    numerical derivatives (this is extremely cheap compared to other machine
    learning methods). All these simulations are passed through a network
    :math:`f_{{\\bf w}}({\\bf d})` with network parameters :math:`{\\bf w}` to
    obtain network outputs :math:`{\\bf x}^i` and
    :math:`\\{{\\bf x}_{\\alpha^-}^i,{\\bf x}_{\\alpha^+}^i\\}`. These
    perturbed values are combined to obtain

    .. math::
        \\frac{\\partial{{\\bf x}^i}}{\\partial\\theta_\\alpha} =
        \\frac{{\\bf x}_{\\alpha^+}^i - {\\bf x}_{\\alpha^-}^i}
        {\\delta\\theta_\\alpha}

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
    δθ : float(n_params,)
        Size of perturbation to model parameters for the numerical derivative
    fiducial : list of tf.data.Dataset().as_numpy_iterators()
        The simulations generated at the fiducial model parameter values used
        for calculating the covariance of network outputs (for fitting). These
        are served ``n_per_device`` at a time as a numpy iterator from a
        TensorFlow dataset.
    derivative : list of tf.data.Dataset().as_numpy_iterators()
        The simulations generated at parameter values perturbed from the
        fiducial used to calculate the numerical derivative of network outputs
        with respect to model parameters (for fitting).  These are served
        ``n_per_device`` at a time as a numpy iterator from a TensorFlow
        dataset.
    validation_fiducial : list of tf.data.Dataset().as_numpy_iterators()
        The simulations generated at the fiducial model parameter values used
        for calculating the covariance of network outputs (for validation).
        These are served ``n_per_device`` at a time as a numpy iterator from a
        TensorFlow dataset.
    validation_derivative : list of tf.data.Dataset().as_numpy_iterators()
        The simulations generated at parameter values perturbed from the
        fiducial used to calculate the numerical derivative of network outputs
        with respect to model parameters (for validation).  These are served
        ``n_per_device`` at a time as a numpy iterator from a TensorFlow
        dataset.
    fiducial_iterations : int
        The number of iterations over the fiducial dataset
    derivative_iterations : int
        The number of iterations over the derivative dataset
    derivative_output_shape : tuple
        The shape of the output of the derivatives from the network
    fiducial_batch_shape : tuple
        The shape of each batch of fiducial simulations (without input or
        summary shape)
    derivative_batch_shape : tuple
        The shape of each batch of derivative simulations (without input or
        summary shape)
    """
    def __init__(self, n_s, n_d, n_params, n_summaries, input_shape, θ_fid,
                 model, optimiser, key_or_state, fiducial, derivative, δθ,
                 host, devices, n_per_device, validation_fiducial=None,
                 validation_derivative=None, prefetch=None,
                 cache=False):
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
        derivative : float(n_d, 2, n_params, input_shape)
            The simulations generated at parameter values perturbed from the
            fiducial used to calculate the numerical derivative of network
            outputs with respect to model parameters (for fitting)
        δθ : float(n_params,)
            Size of perturbation to model parameters for the numerical
            derivative
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
        validation_derivative : float(n_d, 2, n_params, input_shape) or None
            The simulations generated at parameter values perturbed from the
            fiducial used to calculate the numerical derivative of network
            outputs with respect to model parameters (for validation)
        prefetch : tf.data.AUTOTUNE or int or None, default=None
            How many simulation to prefetch in the tensorflow dataset
        cache : bool, default=False
            Whether to cache simulations in the tensorflow datasets
        """
        NumericalGradientIMNN.__init__(
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
            δθ=δθ,
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
        """
        self.fiducial_iterations = self.n_s \
            // (self.n_devices * self.n_per_device)
        self.derivative_iterations = self.n_d * 2 * self.n_params \
            // (self.n_devices * self.n_per_device)
        self.derivative_output_shape = (
            self.n_d, 2, self.n_params, self.n_summaries)
        self.fiducial_batch_shape = (
            self.n_devices,
            self.fiducial_iterations,
            self.n_per_device)
        self.derivative_batch_shape = (
            self.n_devices,
            self.derivative_iterations,
            self.n_per_device)

    def _set_dataset(self, prefetch, cache):
        """ Transforms the data into lists of tensorflow dataset iterators

        Parameters
        ----------
        prefetch : tf.data.AUTOTUNE or int or None
            How many simulation to prefetch in the tensorflow dataset
        cache : bool
            Whether to cache simulations in the tensorflow datasets

        Raises
        ------
        ValueError
            If ``cache`` and/or ``prefetch`` is None
        TypeError
            If ``cache`` and/or ``prefetch`` is wrong type
        """
        cache = _check_boolean(cache, "cache")
        prefetch = _check_type(prefetch, int, "prefetch", allow_None=True)
        self.fiducial = self.fiducial.reshape(
            self.fiducial_batch_shape + self.input_shape)
        self.fiducial = [
            tf.data.Dataset.from_tensor_slices(
                fiducial)
            for fiducial in self.fiducial]

        self.derivative = self.derivative.reshape(
            self.derivative_batch_shape + self.input_shape)
        self.derivative = [
            tf.data.Dataset.from_tensor_slices(
                derivative)
            for derivative in self.derivative]

        if cache:
            self.fiducial = [
                fiducial.cache()
                for fiducial in self.fiducial]
            self.derivative = [
                derivative.cache()
                for derivative in self.derivative]
        if prefetch is not None:
            self.fiducial = [
                fiducial.prefetch(prefetch)
                for fiducial in self.fiducial]
            self.derivative = [
                derivative.prefetch(prefetch)
                for derivative in self.derivative]

        self.fiducial = [
            fiducial.repeat().as_numpy_iterator()
            for fiducial in self.fiducial]
        self.derivative = [
            derivative.repeat().as_numpy_iterator()
            for derivative in self.derivative]

        if self.validate:
            self.validation_fiducial = self.validation_fiducial.reshape(
                self.fiducial_batch_shape + self.input_shape)
            self.validation_fiducial = [
                tf.data.Dataset.from_tensor_slices(
                    fiducial)
                for fiducial in self.validation_fiducial]

            self.validation_derivative = self.validation_derivative.reshape(
                self.derivative_batch_shape + self.input_shape)
            self.validation_derivative = [
                tf.data.Dataset.from_tensor_slices(
                    derivative)
                for derivative in self.validation_derivative]

            if cache:
                self.validation_fiducial = [
                    fiducial.cache()
                    for fiducial in self.validation_fiducial]
                self.validation_derivative = [
                    derivative.cache()
                    for derivative in self.validation_derivative]
            if prefetch is not None:
                self.validation_fiducial = [
                    fiducial.prefetch(prefetch)
                    for fiducial in self.validation_fiducial]
                self.validation_derivative = [
                    derivative.prefetch(prefetch)
                    for derivative in self.validation_derivative]

            self.validation_fiducial = [
                fiducial.repeat().as_numpy_iterator()
                for fiducial in self.validation_fiducial]
            self.validation_derivative = [
                derivative.repeat().as_numpy_iterator()
                for derivative in self.validation_derivative]

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
        self.batch_gradients = [
            jax.jit(
                partial(
                    self._get_batch_summaries,
                    θ=self.θ_fid,
                    gradient=True,
                    derivative=False),
                device=device)
            for device in self.devices]

    def get_summary(self, inputs, w, θ, derivative=False, gradient=False):
        """ Returns a single summary of a simulation or its gradient

        Parameters
        ----------
        inputs : float(input_shape) or tuple
            A single simulation to pass through the network or a tuple of
                - **dΛ_dx** *float(input_shape, n_params)* -- the derivative of
                  the loss function with respect to a network summary
                - **d** *float(input_shape)* -- a simulation to compress with
                  the network
        w : list
            The network parameters
        θ : float(n_params,)
            The value of the parameters to generate the simulation at
            (fiducial), unused if not simulating on the fly
        derivative : bool, default=False
            Whether a derivative of the simulation with respect to model
            parameters is also passed. This must be False for
            NumericalGradientIMNN
        gradient : bool, default=False
            Whether to calculate the gradient with respect to model parameters
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
        if gradient:
            dΛ_dx, d = inputs
            dx_dw = jacrev(fn, argnums=1)(d, w)
            return self._construct_gradient(dx_dw, aux=dΛ_dx, func="einsum")
        else:
            return fn(inputs, w)

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
            The iterators for fiducial simulations for fitting or validation
        list of tf.data.Dataset().as_numpy_iterators:
            The iterators for derivative simulations for fitting or validation
        """
        if validate:
            fiducial = self.validation_fiducial
            derivative = self.validation_derivative
        else:
            fiducial = self.fiducial
            derivative = self.derivative
        return fiducial, derivative

    def get_summaries(self, w, key=None, validate=False):
        """Gets all network outputs and derivatives wrt model parameters

        Selects either the fitting or validation sets and loops through the
        iterator on each XLA device to pass them through the network to get the
        network outputs. These are then pushed back to the host for the
        computation of the loss function.

        The fiducial simulations are processed first and then the simulations
        which are varied with respect to model parameters for the derivatives.

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
        float(n_d, 2, n_params, n_summaries):
            The outputs of the network of simulations made at perturbed
            parameter values to construct the derivative of the network outputs
            with respect to the model parameters numerically
        """
        d, d_mp = self._collect_input(key, validate=validate)

        x = [np.zeros((self.fiducial_iterations,
                       self.n_per_device,
                       self.n_summaries))
             for i in range(self.n_devices)]
        x_mp = [np.zeros((self.derivative_iterations,
                          self.n_per_device,
                          self.n_summaries))
                for i in range(self.n_devices)]

        for i in range(self.fiducial_iterations):
            for j, (fn, dataset) in enumerate(zip(self.batch_summaries, d)):
                x[j] = jax.ops.index_update(
                    x[j], jax.ops.index[i], fn(next(dataset), w))

        for i in range(self.derivative_iterations):
            for j, (fn, dataset) in enumerate(zip(self.batch_summaries, d_mp)):
                x_mp[j] = jax.ops.index_update(
                    x_mp[j], jax.ops.index[i], fn(next(dataset), w))

        if self.n_devices > 1:
            x = [jax.device_put(_x, device=self.host)
                 for _x in x]
            x_mp = [jax.device_put(_x_mp, device=self.host)
                    for _x_mp in x_mp]

        x = np.stack(x, 0).reshape((self.n_s, self.n_summaries))
        x_mp = np.stack(x_mp, 0).reshape(self.derivative_output_shape)
        return x, x_mp

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
            - **d2Λ_dxdθ** *float(n_d, 2, n_params, n_summaries)* -- the
              derivative of the loss function with respect to the derivative of
              network summaries with respect to model parameters

        Returns
        -------
        list:
            a list of sets of derivatives of the loss function with respect to
            network summaries placed on each XLA device
        list:
            a list of sets of derivatives of the loss function with respect to
            the derivative of network summaries with respect to model
            parameters
        """
        d2Λ_dxdθ = [jax.device_put(val, device=device) for val, device in zip(
            dΛ_dx[1].reshape(
                self.derivative_batch_shape + (self.n_summaries,)),
            self.devices)]
        dΛ_dx = [jax.device_put(val, device=device) for val, device in zip(
            dΛ_dx[0].reshape(
                self.fiducial_batch_shape + (self.n_summaries,)),
            self.devices)]
        return dΛ_dx, d2Λ_dxdθ

    def get_gradient(self, dΛ_dx, w, key=None):
        """Aggregates gradients together to update the network parameters

        To avoid having to calculate the gradient with respect to all the
        simulations at once we aggregated by addition the gradient calculation
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
            - **d2Λ_dxdθ** *float(n_d, 2, n_params, n_summaries)* -- the
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
        dΛ_dx, dΛ_dx_mp = self._split_dΛ_dx(dΛ_dx)
        d, d_mp = self._collect_input(key, validate=False)

        gradient = [
            jax.jit(
                partial(self._construct_gradient, func="zeros"),
                device=device)(w)
            for device in self.devices]

        for i in range(self.fiducial_iterations):
            for j, (fn, dataset, grad, device) in enumerate(
                    zip(self.batch_gradients, d, dΛ_dx, self.devices)):
                gradient[j] = jax.jit(add_nested_pytrees, device=device)(
                    gradient[j],
                    fn((grad[i], next(dataset)), w))

        for i in range(self.derivative_iterations):
            for j, (fn, dataset, grad, device) in enumerate(
                    zip(self.batch_gradients, d_mp, dΛ_dx_mp, self.devices)):
                gradient[j] = jax.jit(add_nested_pytrees, device=device)(
                    gradient[j],
                    fn((grad[i], next(dataset)), w))

        return list(jax.jit(add_nested_pytrees, device=self.host)(*gradient))
