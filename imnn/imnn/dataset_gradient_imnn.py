from imnn.imnn import AggregatedGradientIMNN
from imnn.utils.utils import _check_type


class DatasetGradientIMNN(AggregatedGradientIMNN):
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

    In DatasetGradientIMNN the input datasets should be lists of ``n_devices``
    ``tf.data.Datasets``. Please note, due to the many various ways of
    constructing datasets to load data, there is no checking and any improperly
    made dataset will either fail (best result) or provide the wrong result
    (worst case scenario!). For this reason it is advised to use
    :func:`~AggregatedGradientIMNN` if data will fit into CPU memory at least.
    If not, the next safest way is to construct a set of TFRecords and
    construct the dataset from that.

    Examples
    --------
    Here are various ways to construct the datasets for passing to the
    :func:~`DatasetGradientIMNN`. Note that these are not the only
    ways, but they should give something to follow to generate your own
    datasets. First we'll generate some data (just random noise with zero mean
    and unit variance). We'll generate 1000 simulations at the fiducial and
    we'll use jax to calculate the derivatives with respect to the mean and
    variance for 100 of these. We'll save each of these simulations into its
    own individual file (named by seed value).

    .. code-block:: python

        import glob
        import jax
        import jax.numpy as np
        import tensorflow as tf
        from imnn import TFRecords
        from functools import partial
        from imnn.utils import value_and_jacfwd

        n_s = 1000
        n_d = 100
        n_params = 2
        input_shape = (10,)

        def simulator(key, θ):
            return θ[0] + (jax.random.normal(key, shape=input_shape)
                * np.sqrt(θ[1]))

        θ_fid = np.array([0., 1.])

        get_sims_and_ders = value_and_jacfwd(simulator, argnums=1)

        rng = jax.random.PRNGKey(0)
        rng, data_key = jax.random.split(rng)
        data_keys = np.array(jax.random.split(rng, num=2 * n_s))

        fiducial, derivative = jax.vmap(get_sims_and_ders)(
            data_keys[:n_d], np.repeat(np.expand_dims(θ_fid, 0), n_d, axis=0))

        remaining = jax.vmap(simulator)(
            data_keys[n_d:n_s],
            np.repeat(np.expand_dims(θ_fid, 0), n_s - n_d, axis=0))

        validation_fiducial, validation_derivative = jax.vmap(
            get_sims_and_ders)(
                data_keys[n_s:n_s + n_d],
                np.repeat(np.expand_dims(θ_fid, 0), n_d, axis=0))

        validation_remaining = jax.vmap(simulator)(
            data_keys[n_s + n_d:],
            np.repeat(np.expand_dims(θ_fid, 0), n_s - n_d, axis=0))

        for i, (simulation, validation_simulation) in enumerate(zip(
                fiducial, validation_fiducial)):
            np.save(f"tmp/fiducial_{i:04d}.npy", simulation)
            np.save(f"tmp/validation_fiducial_{i:04d}.npy",
                    validation_simulation)

        for i, (simulation, validation_simulation) in enumerate(zip(
                derivative, validation_derivative)):
            np.save(f"tmp/derivative_{i:04d}.npy", simulation)
            np.save(f"tmp/validation_derivative_{i:04d}.npy",
                    validation_simulation)

    Now we'll define how many devices to farm out our calculations to. We need
    to know this because we want to make a separate dataset for each device.
    We'll also set the number of simulations which can be processed at once on
    each device, this should be as high as possible without running out of
    memory on any individual device for quickest fitting

    .. code-block:: python

        devices = jax.devices("gpu")
        n_devices = len(devices)
        n_per_device = 100

    To best accelerate the aggregation of the gradient calculation the
    computation is split into two parts, a ``main`` loop which loops through
    ``n_d`` simulation with its derivative with respect to model parameters,
    and a ``remaining`` loop of ``n_s - n_d`` iterations, where just
    simulations are looped through to calculate any other necessary summaries
    to estimate the covariance. Note this is true even if ``n_s = n_s`` the
    remaining loop just has zero iterations. So to construct the dataset define
    the shapes for the data to be reshaped into for proper construction of the
    datasets to be used when fitting the IMNN.

    .. code-block:: python

        batch_shape = (
            n_devices,
            n_d // (n_devices * n_per_device),
            n_per_device) + input_shape

        remaining_batch_shape = (
            n_devices,
            (n_s - n_d) // (n_devices * n_per_device),
            n_per_device) + input_shape

    The simplest way to construct a dataset is simply using the numpy arrays
    from memory (note if you're going to do this you should really just use
    ``AggregatedGradientIMNN``, its more or less the same!), i.e.

    .. code-block:: python

        main = [
            tf.data.Dataset.from_tensor_slices(
                (fiducial, derivative)).repeat().as_numpy_iterator()
            for fiducial, derivative in zip(
                fiducial.reshape(batch_shape),
                derivative.reshape(batch_shape + (n_params,)))]

        remaining = [
            tf.data.Dataset.from_tensor_slices(fiducial
                ).repeat().as_numpy_iterator()
            for fiducial in remaining.reshape(
                remaining_batch_shape)]

        validation_main = [
            tf.data.Dataset.from_tensor_slices(
                (fiducial, derivative)).repeat().as_numpy_iterator()
            for fiducial, derivative in zip(
                validation_fiducial.reshape(batch_shape),
                validation_derivative.reshape(batch_shape + (n_params,)))]

        validation_remaining = [
            tf.data.Dataset.from_tensor_slices(fiducial
                ).repeat().as_numpy_iterator()
            for fiducial in validation_remaining.reshape(
                remaining_batch_shape)]

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

        main = [
            tf.data.Dataset.zip((
                 tf.data.Dataset.from_generator(
                     partial(
                         generator,
                         "tmp",
                         "fiducial",
                         n_d),
                     tf.float32),
                tf.data.Dataset.from_generator(
                     partial(
                         generator,
                         "tmp",
                         "derivative",
                         n_d),
                     tf.float32))
                ).take(n_d // n_devices
                ).batch(n_per_device
                ).repeat(
                ).as_numpy_iterator()
            for _ in range(n_devices)]

        remaining = [
            tf.data.Dataset.from_generator(
                partial(
                    generator,
                    "tmp",
                    "remaining",
                    n_s - n_d),
                tf.float32
                ).take((n_s - n_d) // n_devices
                ).batch(n_per_device
                ).repeat(
                ).as_numpy_iterator()
            for _ in range(n_devices)]

        validation_main = [
            tf.data.Dataset.zip((
                 tf.data.Dataset.from_generator(
                     partial(
                         generator,
                         "tmp",
                         "validation_fiducial",
                         n_d),
                     tf.float32),
                tf.data.Dataset.from_generator(
                     partial(
                         generator,
                         "tmp",
                         "validation_derivative",
                         n_d),
                     tf.float32))
                ).take(n_d // n_devices
                ).batch(n_per_device
                ).repeat(
                ).as_numpy_iterator()
            for _ in range(n_devices)]

        validation_remaining = [
            tf.data.Dataset.from_generator(
                partial(
                    generator,
                    "tmp",
                    "validation_remaining",
                    n_s - n_d),
                tf.float32
                ).take((n_s - n_d) // n_devices
                ).batch(n_per_device
                ).repeat(
                ).as_numpy_iterator()
            for _ in range(n_devices)]

    The datasets must be built exactly like this, with the taking and batching
    and repeating. The zipping of the main datasets is equal as important to
    pass both the fiducial simulation and its derivative at once, which is
    needed to calculate the final gradient. To prefetch and cache the
    loaded files we can add the extra steps in the datasets, e.g.

    .. code-block:: python

        main = [
            tf.data.Dataset.zip((
                 tf.data.Dataset.from_generator(
                     partial(
                         generator,
                         "tmp",
                         "fiducial",
                         n_d),
                     tf.float32),
                tf.data.Dataset.from_generator(
                     partial(
                         generator,
                         "tmp",
                         "derivative",
                         n_d),
                     tf.float32))
                ).take(n_d // n_devices
                ).batch(n_per_device
                ).cache(
                ).prefetch(tf.data.AUTOTUNE
                ).repeat(
                ).as_numpy_iterator()

    etc.

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

        def get_simulation(seed, directory=None, filename=None):
            return np.load(f"{directory}/{filename}_{seed:04d}.npy")

        writer.write_record(
            n_sims=n_d,
            get_simulation=lambda seed: get_simulation(
                seed, directory="tmp", filename="fiducial"),
            directory="tmp",
            filename="fiducial")

        writer.write_record(
            n_sims=n_s - n_d,
            get_simulation=lambda seed: get_simulation(
                seed, directory="tmp", filename="remaining"),
            directory="tmp",
            filename="remaining")

        writer.write_record(
            n_sims=n_d,
            get_simulation=lambda seed: get_simulation(
                seed, directory="tmp", filename="derivative"),
            directory="tmp",
            filename="derivative")

        writer.write_record(
            n_sims=n_d,
            get_simulation=lambda seed: get_simulation(
                seed, directory="tmp", filename="validation_fiducial"),
            directory="tmp",
            filename="validation_fiducial")

        writer.write_record(
            n_sims=n_s - n_d,
            get_simulation=lambda seed: get_simulation(
                seed, directory="tmp", filename="validation_remaining"),
            directory="tmp",
            filename="validation_remaining")

        writer.write_record(
            n_sims=n_d,
            get_simulation=lambda seed: get_simulation(
                seed, directory="tmp", filename="validation_derivative"),
            directory="tmp",
            filename="validation_derivative")

    We can then read these to a dataset using the parser from the TFRecords
    class mapping the format of the data to a 32-bit float

    .. code-block:: python

        fiducial = [
            tf.data.TFRecordDataset(
                    sorted(glob.glob("tmp/fiducial_*.tfrecords")),
                    num_parallel_reads=1
                ).map(writer.parser
                ).skip(i * n_s // n_devices
                ).take(n_s // n_devices)
            for i in range(n_devices)]

        main = [
            tf.data.Dataset.zip((
                fiducial[i],
                tf.data.TFRecordDataset(
                    sorted(glob.glob("tmp/derivative_*.tfrecords")),
                    num_parallel_reads=1).map(
                        lambda example: writer.derivative_parser(
                            example, n_params=n_params)))
                ).take(n_d // n_devices
                ).batch(n_per_device
                ).repeat(
                ).as_numpy_iterator()
            for i in range(n_devices)]

        remaining = [
            fiducial[i].skip(n_d // n_devices
                ).take((n_s - n_d) // n_devices
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
                ).take(n_s // n_devices)
            for i in range(n_devices)]

        validation_main = [
            tf.data.Dataset.zip((
                validation_fiducial[i],
                tf.data.TFRecordDataset(
                    sorted(glob.glob("tmp/validation_derivative_*.tfrecords")),
                    num_parallel_reads=1).map(
                        lambda example: writer.derivative_parser(
                            example, n_params=n_params)))
                ).take(n_d // n_devices
                ).batch(n_per_device
                ).repeat(
                ).as_numpy_iterator()
            for i in range(n_devices)]

        validation_remaining = [
            validation_fiducial[i].skip(n_d // n_devices
                ).take((n_s - n_d) // n_devices
                ).batch(n_per_device
                ).repeat(
                ).as_numpy_iterator()
            for i in range(n_devices)]

    Parameters
    ----------
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
                 model, optimiser, key_or_state, main, remaining, host,
                 devices, n_per_device, validation_main=None,
                 validation_remaining=None):
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
        main : list of tf.data.Dataset().as_numpy_iterators()
            The simulations generated at the fiducial model parameter values
            used for calculating the covariance of network outputs and their
            derivatives with respect to the physical model parameters (for
            fitting). These are served ``n_per_device`` at at time as a numpy
            iterator from a TensorFlow dataset.
        remaining : list of tf.data.Dataset().as_numpy_iterators()
            The ``n_s - n_d`` simulations generated at the fiducial model
            parameter values used for calculating the covariance of network
            outputs with a derivative counterpart (for fitting). These are
            served ``n_per_device`` at at time as a numpy iterator from a
            TensorFlow dataset.
        host: jax.device
            The main device where the Fisher calculation is performed
        devices: list
            A list of the available jax devices (from ``jax.devices()``)
        n_per_device: int
            Number of simulations to handle at once, this should be as large as
            possible without letting the memory overflow for the best
            performance
        validation_main : list of tf.data.Dataset().as_numpy_iterators()
            The simulations generated at the fiducial model parameter values
            used for calculating the covariance of network outputs and their
            derivatives with respect to the physical model parameters
            (for validation). These are served ``n_per_device`` at at time as a
            numpy iterator from a TensorFlow dataset.
        validation_remaining : list of tf.data.Dataset().as_numpy_iterators()
            The ``n_s - n_d`` simulations generated at the fiducial model
            parameter values used for calculating the covariance of network
            outputs with a derivative counterpart (for validation). Served
            ``n_per_device`` at time as a numpy iterator from a TensorFlow
            dataset.
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
            fiducial=None,
            derivative=None,
            validation_fiducial=None,
            validation_derivative=None,
            host=host,
            devices=devices,
            n_per_device=n_per_device)
        self._set_prebuilt_dataset(
            main, remaining, validation_main, validation_remaining)

    def _set_data(self, fiducial, derivative, validation_fiducial,
                  validation_derivative):
        """Overwritten function to prevent setting fiducial attributes

        Parameters
        ----------
        fiducial : None
        derivative : None
        validation_fiducial : None
        validation_derivative : None
        """
        pass

    def _set_dataset(self, prefetch=None, cache=None):
        """Overwritten function to prevent building dataset, does list check

        Parameters
        ----------
        prefetch : None
        cache : None
        """
        pass

    def _set_prebuilt_dataset(
            self, main, remaining, validation_main, validation_remaining):
        """Set preconstructed dataset iterators

        Parameters
        ----------
        main : list of tf.data.Dataset().as_numpy_iterators()
            The simulations generated at the fiducial model parameter values
            used for calculating the covariance of network outputs and their
            derivatives with respect to the physical model parameters (for
            fitting). These are served ``n_per_device`` at at time as a numpy
            iterator from a TensorFlow dataset.
        remaining : list of tf.data.Dataset().as_numpy_iterators()
            The ``n_s - n_d`` simulations generated at the fiducial model
            parameter values used for calculating the covariance of network
            outputs with a derivative counterpart (for fitting). These are
            served ``n_per_device`` at at time as a numpy iterator from a
            TensorFlow dataset.
        validation_main : list of tf.data.Dataset().as_numpy_iterators()
            The simulations generated at the fiducial model parameter values
            used for calculating the covariance of network outputs and their
            derivatives with respect to the physical model parameters
            (for validation). These are served ``n_per_device`` at at time as a
            numpy iterator from a TensorFlow dataset.
        validation_remaining : list of tf.data.Dataset().as_numpy_iterators()
            The ``n_s - n_d`` simulations generated at the fiducial model
            parameter values used for calculating the covariance of network
            outputs with a derivative counterpart (for validation). Served
            ``n_per_device`` at time as a numpy iterator from a TensorFlow
            dataset.

        Raises
        ------
        ValueError
            if main or remaining are None
        ValueError
            if length of any input list is not equal to number of devices
        TypeError
            if any input is not a list
        """
        self.main = _check_type(main, list, "main", shape=self.n_devices)
        self.remaining = _check_type(
            remaining, list, "remaining", shape=self.n_devices)
        if ((validation_main is not None)
                and (validation_remaining is not None)):
            self.validation_main = _check_type(
                validation_main, list, "validation_main", shape=self.n_devices)
            self.validation_remaining = _check_type(
                validation_remaining, list, "validation_remaining",
                shape=self.n_devices)
            self.validate = True
