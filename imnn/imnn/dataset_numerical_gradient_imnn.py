import numpy as np
from imnn.imnn import AggregatedNumericalGradientIMNN
from imnn.utils.utils import _check_input, _check_type


class DatasetNumericalGradientIMNN(AggregatedNumericalGradientIMNN):
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

    In DatasetNumericalGradientIMNN the input datasets should be lists of
    ``n_devices`` ``tf.data.Datasets``. Please note, due to the many various
    ways of constructing datasets to load data, there is no checking and any
    improperly made dataset will either fail (best result) or provide the wrong
    result (worst case scenario!). For this reason it is advised to use
    :func:`~AggregatedNumericalGradientIMNN` if data will fit into CPU memory
    at least. If not, the next safest way is to construct a set of TFRecords
    and construct the dataset from that.

    Examples
    --------
    Here are various ways to construct the datasets for passing to the
    :func:~`DatasetNumericalGradientIMNN`. Note that these are not the only
    ways, but they should give something to follow to generate your own
    datasets. First we'll generate some data (just random noise with zero mean
    and unit variance) and perturb the mean and the variance of this noise to
    calculate numerical derivatives with respect to the model parameters.
    We'll generate 1000 simulations at the fiducial and 100 for each parameter
    varied above and below the fiducial. We'll save each of these simulations
    into its own individual file (named by seed value).

    .. code-block:: python

        import glob
        import jax
        import jax.numpy as np
        import tensorflow as tf
        from imnn import TFRecords
        from functools import partial

        n_s = 1000
        n_d = 100
        n_params = 2
        input_shape = (10,)

        def simulator(key, θ):
            return θ[0] + (jax.random.normal(key, shape=input_shape)
                * np.sqrt(θ[1]))

        θ_fid = np.array([0., 1.])
        δθ = np.array([0.1, 0.1])
        θ_der = (θ_fid
            + np.einsum(
                "i,jk->ijk",
                np.array([-1., 1.]),
                np.diag(δθ)
            / 2.)).reshape((-1, 2))

        rng = jax.random.PRNGKey(0)
        rng, data_key = jax.random.split(rng)
        data_keys = np.array(jax.random.split(rng, num=2 * n_s))

        fiducial = jax.vmap(simulator)(
            data_keys[:n_s],
            np.repeat(np.expand_dims(θ_fid, 0), n_s, axis=0))

        validation_fiducial = jax.vmap(simulator)(
            data_keys[n_s:],
            np.repeat(np.expand_dims(θ_fid, 0), n_s, axis=0))

        numerical_derivative = jax.vmap(simulator)(
            np.repeat(data_keys[:n_d], θ_der.shape[0], axis=0),
            np.tile(θ_der, (n_d, 1))).reshape(
                (n_d, 2, n_params) + input_shape)

        validation_numerical_derivative = jax.vmap(simulator)(
            np.repeat(data_keys[n_s:n_d + n_s], θ_der.shape[0], axis=0),
            np.tile(θ_der, (n_d, 1))).reshape(
                (n_d, 2, n_params) + input_shape)

        for i, (simulation, validation_simulation) in enumerate(
                zip(fiducial, validation_fiducial)):
            np.save(f"tmp/fiducial_{i:04d}.npy", simulation)
            np.save(f"tmp/validation_fiducial_{i:04d}.npy",
                    validation_simulation)

        for i, (simulation, validation_simulation) in enumerate(
                zip(numerical_derivative, validation_numerical_derivative)):
            np.save(f"tmp/numerical_derivative_{i:04d}.npy", simulation)
            np.save(f"tmp/validation_numerical_derivative_{i:04d}.npy",
                    validation_simulation

    Now we'll define how many devices to farm out our calculations to. We need
    to know this because we want to make a separate dataset for each device.
    We'll also set the number of simulations which can be processed at once on
    each device, this should be as high as possible without running out of
    memory on any individual device for quickest fitting

    .. code-block:: python

        devices = jax.devices("gpu")
        n_devices = len(devices)
        n_per_device = 100

    Using this we can define the shapes for the data to be reshaped into for
    proper construction of the datasets to be used when fitting the IMNN.

    .. code-block:: python

        fiducial_shape = (
            n_devices,
            n_s // (n_devices * n_per_device),
            n_per_device) + input_shape

        derivative_shape = (
            n_devices,
            2 * n_params * n_d // (n_devices * n_per_device),
            n_per_device) + input_shape

    The simplest way to construct a dataset is simply using the numpy arrays
    from memory (note if you're going to do this you should really just use
    ``AggregatedNumericalGradientIMNN``, its more or less the same!), i.e.

    .. code-block:: python

        fiducial = [
            tf.data.Dataset.from_tensor_slices(
                fid).repeat().as_numpy_iterator()
            for fid in fiducial.reshape(fiducial_shape)]

        numerical_derivative = [
            tf.data.Dataset.from_tensor_slices(
                der).repeat().as_numpy_iterator()
            for der in numerical_derivative.reshape(derivative_shape)]

        validation_fiducial = [
            tf.data.Dataset.from_tensor_slices(
                fid).repeat().as_numpy_iterator()
            for fid in validation_fiducial.reshape(fiducial_shape)]

        validation_numerical_derivative = [
            tf.data.Dataset.from_tensor_slices(
                der).repeat().as_numpy_iterator()
            for der in validation_numerical_derivative.reshape(
                derivative_shape)]

    However, if the data is too large to fit in memory then we can use the
    npy files that we saved by loading them via a generator

    .. code-block:: python

        def generator(directory, filename, total):
            i = 0
            while i < total:
                yield np.load(f"{directory}/{filename}_{i:04d}.npy")
                i += 1

    We can then build the datasets like:

    .. code-block:: python

        fiducial = [
            tf.data.Dataset.from_generator(
                partial(
                    generator,
                    "tmp",
                    "fiducial",
                    n_s),
                tf.float32
                ).take(n_s // n_devices
                ).batch(n_per_device
                ).repeat(
                ).as_numpy_iterator()
            for _ in range(n_devices)]

        numerical_derivative = [
            tf.data.Dataset.from_generator(
                partial(
                    generator,
                    "tmp",
                    "numerical_derivative",
                    n_d),
                tf.float32
                ).flat_map(
                    lambda x: tf.data.Dataset.from_tensor_slices(x)
                ).flat_map(
                    lambda x: tf.data.Dataset.from_tensor_slices(x)
                ).take(2 * n_params * n_d // n_devices
                ).batch(n_per_device
                ).repeat(
                ).as_numpy_iterator()
            for _ in range(n_devices)]

        validation_fiducial = [
            tf.data.Dataset.from_generator(
                partial(
                    generator,
                    "tmp",
                    "validation_fiducial",
                    n_s),
                tf.float32
                ).take(n_s // n_devices
                ).batch(n_per_device
                ).repeat(
                ).as_numpy_iterator()
            for _ in range(n_devices)]

        validation_numerical_derivative = [
            tf.data.Dataset.from_generator(
                partial(
                    generator,
                    "tmp",
                    "validation_numerical_derivative",
                    n_d),
                tf.float32
                ).flat_map(
                    lambda x: tf.data.Dataset.from_tensor_slices(x)
                ).flat_map(
                    lambda x: tf.data.Dataset.from_tensor_slices(x)
                ).take(2 * n_params * n_d // n_devices
                ).batch(n_per_device
                ).repeat(
                ).as_numpy_iterator()
            for _ in range(n_devices)]

    The datasets must be built exactly like this, with the taking and batching
    and repeating. Importantly both the ``flat_map`` over the datasets for the
    numerical derivatives are needed to unwrap the perturbation direction and
    the parameter direction in each numpy file. To prefetch and cache the
    loaded files we can add the extra steps in the datasets, e.g.

    .. code-block:: python

        fiducial = [
            tf.data.Dataset.from_generator(
                partial(
                    generator,
                    "tmp",
                    "fiducial",
                    n_s),
                tf.float32
                ).take(n_s // n_devices
                ).batch(n_per_device
                ).cache(
                ).prefetch(tf.data.AUTOTUNE
                ).repeat(
                ).as_numpy_iterator()
            for _ in range(n_devices)]

    This loading will be quite slow because the files need to be opened each
    time, but we can build TFRecords which are quicker to load. There is a
    writer able to do the correct format. The TFRecords should be a couple
    hundred Mb for best flow-through, so we can keep filling the record until
    this size is reached.

    .. code-block:: python

        record_size = 200 #Mb
        writer = TFRecords.TFRecords(record_size=record_size)

    We need a function which grabs single simulations from an array (or file)
    to add to the record

    .. code-block:: python

        def get_fiducial(seed, directory=None, filename=None):
            return np.load(f"{directory}/{filename}_{seed:04d}.npy")

        def get_derivative(seed, der, params, directory=None, filename=None):
            return np.load(
                f"{directory}/{filename}_{seed:04d}.npy")[der, params]

        writer.write_record(
            n_sims=n_s,
            get_simulation=lambda seed: get_fiducial(
                seed, directory="tmp", filename="fiducial"),
            fiducial=True,
            directory="tmp",
            filename="fiducial")

        writer.write_record(
            n_sims=n_d,
            get_simulation=lambda seed, der, param: get_derivative(
                seed, der, param, directory="tmp",
                filename="numerical_derivative"),
            fiducial=False,
            n_params=n_params,
            directory="tmp",
            filename="numerical_derivative")

        writer.write_record(
            n_sims=n_s,
                get_simulation=lambda seed: get_fiducial(
                seed, directory="tmp",
                filename="validation_fiducial"),
            fiducial=True,
            directory="tmp",
            filename="validation_fiducial")

        writer.write_record(
            n_sims=n_d,
            get_simulation=lambda seed, der, param: get_derivative(
                seed, der, param, directory="tmp",
                filename="validation_numerical_derivative"),
            fiducial=False,
            n_params=n_params,
            directory="tmp",
            filename="validation_numerical_derivative")

    We can then read these to a dataset using the parser from the TFRecords
    class mapping the format of the data to a 32-bit float

    .. code-block:: python

        fiducial = [
            tf.data.TFRecordDataset(
                    sorted(glob.glob("tmp/fiducial_*.tfrecords")),
                    num_parallel_reads=1
                ).map(writer.parser
                ).skip(i * n_s // n_devices
                ).take(n_s // n_devices
                ).batch(n_per_device
                ).repeat(
                ).as_numpy_iterator()
            for i in range(n_devices)]

        numerical_derivative = [
            tf.data.TFRecordDataset(
                    sorted(glob.glob("tmp/numerical_derivative_*.tfrecords")),
                    num_parallel_reads=1
                ).map(writer.parser
                ).skip(i * 2 * n_params * n_d // n_devices
                ).take(2 * n_params * n_d // n_devices
                ).batch(n_per_device
                ).repeat(
                ).as_numpy_iterator()
            for i in range(n_devices)]

        validation_fiducial = [
            tf.data.TFRecordDataset(
                    sorted(glob.glob("tmp/validation_fiducial_*.tfrecords")),
                    num_parallel_reads=1
                ).map(writer.parser
                ).skip(i * n_s // n_devices
                ).take(n_s // n_devices
                ).batch(n_per_device
                ).repeat(
                ).as_numpy_iterator()
            for _ in range(n_devices)]

        validation_numerical_derivative = [
            tf.data.TFRecordDataset(
                    sorted(glob.glob(
                        "tmp/validation_numerical_derivative_*.tfrecords")),
                    num_parallel_reads=1
                ).map(writer.parser
                ).skip(i * 2 * n_params * n_d // n_devices
                ).take(2 * n_params * n_d // n_devices
                ).batch(n_per_device
                ).repeat(
                ).as_numpy_iterator()
            for _ in range(n_devices)]

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
                 validation_derivative=None):
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
        fiducial : list of tf.data.Dataset()
            The simulations generated at the fiducial model parameter values
            used for calculating the covariance of network outputs (for
            fitting). These are served ``n_per_device`` at at time as a numpy
            iterator from a TensorFlow dataset.
        derivative : list of tf.data.Dataset()
            The simulations generated at parameter values perturbed from the
            fiducial used to calculate the numerical derivative of network
            outputs with respect to model parameters (for fitting).  These are
            served ``n_per_device`` at at time as a numpy iterator from a
            TensorFlow dataset.
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
        validation_fiducial : list of tf.data.Dataset()
            The simulations generated at the fiducial model parameter values
            used for calculating the covariance of network outputs (for
            validation). These are served ``n_per_device`` at at time as a
            numpy iterator from a TensorFlow dataset.
        validation_derivative : list of tf.data.Dataset()
            The simulations generated at parameter values perturbed from the
            fiducial used to calculate the numerical derivative of network
            outputs with respect to model parameters (for validation).  These
            are served ``n_per_device`` at at time as a numpy iterator from a
            TensorFlow dataset.
        """
        super().__init__(
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
            validation_derivative=validation_derivative,
            host=host,
            devices=devices,
            n_per_device=n_per_device)

    def _set_data(self, δθ, fiducial, derivative, validation_fiducial,
                  validation_derivative):
        """Checks and sets data attributes with the correct shape

        Parameters
        ----------
        δθ : float(n_params,)
            Size of perturbation to model parameters for the numerical
            derivative
        fiducial : list of tf.data.Datesets
            The simulations generated at the fiducial model parameter values
            used for calculating the covariance of network outputs
            (for fitting)
        derivative : list of tf.data.Datesets
            The derivative of the simulations with respect to the model
            parameters (for fitting)
        validation_fiducial : list of tf.data.Datesets or None, default=None
            The simulations generated at the fiducial model parameter values
            used for calculating the covariance of network outputs
            (for validation). Sets ``validate = True`` attribute if provided
        validation_derivative : list of tf.data.Datesets or None, default=None
            The derivative of the simulations with respect to the model
            parameters (for validation). Sets ``validate = True`` attribute if
            provided

        Raises
        ------
        ValueError
            if δθ is None
        ValueError
            if δθ has wrong shape
        TypeError
            if δθ has wrong type

        Notes
        -----
        No checking is done on the correctness of the tf.data.Dataset
        """
        self.δθ = np.expand_dims(
            _check_input(δθ, (self.n_params,), "δθ"), (0, 1))
        self.fiducial = fiducial
        self.derivative = derivative
        if ((validation_fiducial is not None)
                and (validation_derivative is not None)):
            self.validation_fiducial = validation_fiducial
            self.validation_derivative = validation_derivative
            self.validate = True

    def _set_dataset(self, prefetch=None, cache=None):
        """Overwritten function to prevent building dataset, does list check

        Raises
        ------
        ValueError
            if fiducial or derivative are None
        ValueError
            if any dataset has wrong shape
        TypeError
            if any dataset has wrong type
        """
        _check_type(self.fiducial, list, "fiducial", shape=self.n_devices)
        _check_type(self.derivative, list, "derivative", shape=self.n_devices)
        if self.validate:
            _check_type(self.validation_fiducial, list, "validation_fiducial",
                        shape=self.n_devices)
            _check_type(self.validation_derivative, list,
                        "validation_derivative", shape=self.n_devices)
