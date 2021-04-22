Using the DatasetNumericalGradientIMNN
======================================

When we do not have a generative model which we can automatically or
analytically calculate the derivatives of some simulations we have to
resort to numerical methods to calculate the derivatives of the network
outputs with respect to the physical model parameters necessary to fit
an IMNN. To do this we will use seed-matched derivatives,
i.e. simulations will be made at parameter values above and below the
fiducial for each parameter respectively and using the same seed for
each variation within the pairs. It is important to use seed matching
because this way there is an exact way to separate the effect of
realisation from parameters - many more realisations will be needed and
fitting will be much more difficult without seed matching. Furthermore,
if the simulations are too numerous or too large to fit into memory or
could be accelerated over several devices, we can aggregate the
gradients too. This is an expensive operation and should only be used if
memory over a single device is really an issue. Note that with a fixed
data set for training an IMNN it is important to have a validation set
for early stopping. This is because, with a limited training set there
*will* be accidental correlations which look like they are due to
parameters and the IMNN *will* extract these features. Using a
validation set for early stopping makes sure that once all the features
in the validation set have been extracted then no extra information can
be incorrectly processed.

For this example we are going to summaries the unknown mean,
:math:`\mu`, and variance, :math:`\Sigma`, of :math:`n_{\bf d}=10` data
points of two 1D random Gaussian field,
:math:`{\bf d}=\{d_i\sim\mathcal{N}(\mu,\Sigma)|i\in[1, n_{\bf d}]\}`.
This is an interesting problem since we know the likelihood
analytically, but it is non-Gaussian

.. math:: \mathcal{L}({\bf d}|\mu,\Sigma) = \prod_i^{n_{\bf d}}\frac{1}{\sqrt{2\pi|\Sigma|}}\exp\left[-\frac{1}{2}\frac{(d_i-\mu)^2}{\Sigma}\right]

As well as knowing the likelihood for this problem, we also know what
sufficient statistics describe the mean and variance of the data - they
are the mean and the variance

.. math:: \frac{1}{n_{\bf d}}\sum_i^{n_{\bf d}}d_i = \mu\textrm{  and  }\frac{1}{n_{\bf d}-1}\sum_i^{n_{\bf d}}(d_i-\mu)^2=\Sigma

What makes this an interesting problem for the IMNN is the fact that the
sufficient statistic for the variance is non-linear, i.e. it is a sum of
the square of the data, and so linear methods like MOPED would be lossy
in terms of information.

We can calculate the Fisher information by taking the negative second
derivative of the likelihood taking the expectation by inserting the
relations for the sufficient statistics, i.e. and examining at the
fiducial parameter values

.. math:: {\bf F}_{\alpha\beta} = -\left.\left(\begin{array}{cc}\displaystyle-\frac{n_{\bf d}}{\Sigma}&0\\0&\displaystyle-\frac{n_{\bf d}}{2\Sigma^2}\end{array}\right)\right|_{\Sigma=\Sigma^{\textrm{fid}}}.

Choosing fiducial parameter values of :math:`\mu^\textrm{fid}=0` and
:math:`\Sigma^\textrm{fid}=1` we find that the determinant of the Fisher
information matrix is :math:`|{\bf F}_{\alpha\beta}|=50`.

.. code:: ipython3

    from imnn import DatasetNumericalGradientIMNN

    import jax
    import jax.numpy as np
    from jax.experimental import stax, optimizers

    import tensorflow as tf

We’re going to use 1000 summary vectors, with a length of two, at a time
to make an estimate of the covariance of network outputs and the
derivative of the mean of the network outputs with respect to the two
model parameters.

.. code:: ipython3

    n_s = 1000
    n_d = n_s

    n_params = 2
    n_summaries = n_params

    input_shape = (10,)

The simulator is simply

.. code:: ipython3

    def simulator(key, θ):
        return θ[0] + jax.random.normal(key, shape=input_shape) * np.sqrt(θ[1])

Our fiducial parameter values are :math:`\mu^\textrm{fid}=0` and
:math:`\Sigma^\textrm{fid}=1`. We will vary these values by
:math:`\delta\mu=0.1` and :math:`\delta\Sigma=0.1`.

.. code:: ipython3

    θ_fid = np.array([0., 1.])
    δθ = np.array([0.1, 0.1])
    θ_der = (θ_fid + np.einsum("i,jk->ijk", np.array([-1., 1.]), np.diag(δθ) / 2.)).reshape((-1, 2))

We will use the CPU as the host device and use the GPUs for calculating
the summaries.

.. code:: ipython3

    host = jax.devices("cpu")[0]
    devices = jax.devices("gpu")
    n_devices = len(devices)

Now lets say that we know that we can process 100 simulations at a time
per device before running out of memory, we therefore can set

.. code:: ipython3

    n_per_device = 100

For initialising the neural network a random number generator and we’ll
grab another for generating the data:

.. code:: ipython3

    rng = jax.random.PRNGKey(0)
    rng, model_key, data_key = jax.random.split(rng, num=3)

We’ll make the keys for each of the simulations for fitting and
validation

.. code:: ipython3

    data_keys = np.array(jax.random.split(rng, num=2 * n_s))

.. code:: ipython3

    fiducial = jax.vmap(simulator)(
        data_keys[:n_s],
        np.repeat(np.expand_dims(θ_fid, 0), n_s, axis=0))
    validation_fiducial = jax.vmap(simulator)(
        data_keys[n_s:],
        np.repeat(np.expand_dims(θ_fid, 0), n_s, axis=0))
    numerical_derivative = jax.vmap(simulator)(
        np.repeat(data_keys[:n_s], θ_der.shape[0], axis=0),
        np.tile(θ_der, (n_s, 1))).reshape(
            (n_s, 2, n_params) + input_shape)
    validation_numerical_derivative = jax.vmap(simulator)(
        np.repeat(data_keys[n_s:], θ_der.shape[0], axis=0),
        np.tile(θ_der, (n_s, 1))).reshape(
            (n_s, 2, n_params) + input_shape)

The datasets *must* be made in a very specific way and this is not
currently checked. Any failure to build the dataset in exactly the
correct way will cause either failures or errors in the results. If data
fits in memory then do consider passing the numpy arrays to
``AggregatedNumericalGradientIMNN`` which does all necessary checking.
For the ``DatasetNumericalGradientIMNN`` we need a list of datasets over
each device which output numpy iterators. This means that we need to
reshape the data into the correct shape:

.. code:: ipython3

    fiducial_shape = (
        n_devices,
        n_s // (n_devices * n_per_device),
        n_per_device) + input_shape
    derivative_shape = (
        n_devices,
        2 * n_params * n_d // (n_devices * n_per_device),
        n_per_device) + input_shape

Note that if the reshaping isn’t exact then there will be problems, this
is avoided if passing directly to ``AggregatedNumericalGradientIMNN``,
where checking is automatically done. The datasets then must be made
using:

.. code:: ipython3

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
        for der in validation_numerical_derivative.reshape(derivative_shape)]

Note that if passing a dataset, very flexible data loading can be
performed (as long as it is done carefully). For example, if we saved
each simulation and each set of derivatives to numpy files using

.. code:: python

   for i, (simulation, validation_simulation) in enumerate(zip(
           fiducial, validation_fiducial)):
       np.save(f"tmp/fiducial_{i:04d}.npy", simulation)
       np.save(f"tmp/validation_fiducial_{i:04d}.npy", validation_simulation)

   for i, (simulation, validation_simulation) in enumerate(zip(
           numerical_derivative, validation_numerical_derivative)):
       np.save(f"tmp/numerical_derivative_{i:04d}.npy", simulation)
       np.save(f"tmp/validation_numerical_derivative_{i:04d}.npy",
               validation_simulation)

We could then write the datasets as

.. code:: python

   def generator(directory, filename, total, n_per_device):
       i = 0
       while i < total:
           yield np.load(f"{directory}/{filename}_{i:04d}.npy")
           i += 1

   from functools import partial

   fiducial = [
       tf.data.Dataset.from_generator(
           partial(
               generator,
               "tmp",
               "fiducial",
               n_s,
               n_per_device),
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
               n_d,
               n_per_device),
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
               n_s,
               n_per_device),
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
               n_d,
               n_per_device),
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

Of course we can add ``tf.data.Dataset`` functions like ``prefetch`` and
``cache`` if we want too, i.e. 

.. code:: python

   fiducial = [
       tf.data.Dataset.from_generator(
           partial(
               generator,
               "tmp",
               "fiducial",
               n_s,
               n_per_device),
           tf.float32
           ).take(n_s // n_devices
           ).batch(n_per_device
           ).cache(
           ).prefetch(tf.data.AUTOTUNE
           ).repeat(
           ).as_numpy_iterator()
       for _ in range(n_devices)]

etc.

This loading will be quite slow because the files need to be opened each
time, but we can build TFRecords which are quicker to load. There is a
writer able to do the correct format. The TFRecords should be a couple
hundred Mb for best flow-through, so we can keep filling the record
until this size is reached.

.. code:: python

   from imnn import TFRecords

   record_size = 200 #Mb
   writer = TFRecords(record_size=record_size)

We need a function which grabs single simulations from an array (or
file) to add to the record

.. code:: python

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

We can then read these to a dataset using (note the parser from the
TFRecords class):

.. code:: python

   import glob

   fiducial = [
       tf.data.TFRecordDataset(
               sorted(glob.glob("tmp/fiducial_*.tfrecords")),
               num_parallel_reads=1
           ).map(writer.parser
           ).take(n_s // n_devices
           ).batch(n_per_device
           ).repeat(
           ).as_numpy_iterator()
       for _ in range(n_devices)]

   numerical_derivative = [
       tf.data.TFRecordDataset(
               sorted(glob.glob("tmp/numerical_derivative_*.tfrecords")),
               num_parallel_reads=1
           ).map(writer.parser
           ).take(2 * n_params * n_d // n_devices
           ).batch(n_per_device
           ).repeat(
           ).as_numpy_iterator()
       for _ in range(n_devices)]

   validation_fiducial = [
       tf.data.TFRecordDataset(
               sorted(glob.glob("tmp/validation_fiducial_*.tfrecords")),
               num_parallel_reads=1
           ).map(writer.parser
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
           ).take(2 * n_params * n_d // n_devices
           ).batch(n_per_device
           ).repeat(
           ).as_numpy_iterator()
       for _ in range(n_devices)]

We’re going to use ``jax``\ ’s stax module to build a simple network
with three hidden layers each with 128 neurons and which are activated
by leaky relu before outputting the two summaries. The optimiser will be
a ``jax`` Adam optimiser with a step size of 0.001.

.. code:: ipython3

    model = stax.serial(
        stax.Dense(128),
        stax.LeakyRelu,
        stax.Dense(128),
        stax.LeakyRelu,
        stax.Dense(128),
        stax.LeakyRelu,
        stax.Dense(n_summaries))
    optimiser = optimizers.adam(step_size=1e-3)

The ``DatasetNumericalGradientIMNN`` can now be initialised setting up
the network and the fitting routine (as well as the plotting function)

.. code:: ipython3

    imnn = DatasetNumericalGradientIMNN(
        n_s=n_s, n_d=n_d, n_params=n_params, n_summaries=n_summaries,
        input_shape=input_shape, θ_fid=θ_fid, model=model,
        optimiser=optimiser, key_or_state=model_key, host=host,
        devices=devices, n_per_device=n_per_device, δθ=δθ,
        fiducial=fiducial, derivative=numerical_derivative,
        validation_fiducial=validation_fiducial,
        validation_derivative=validation_numerical_derivative)

To set the scale of the regularisation we use a coupling strength
:math:`\lambda` whose value should mean that the determinant of the
difference between the covariance of network outputs and the identity
matrix is larger than the expected initial value of the determinant of
the Fisher information matrix from the network. How close to the
identity matrix the covariance should be is set by :math:`\epsilon`.
These parameters should not be very important, but they will help with
convergence time.

.. code:: ipython3

    λ = 10.
    ϵ = 0.1

Fitting can then be done simply by calling:

.. code:: ipython3

    imnn.fit(λ, ϵ, patience=10, max_iterations=1000, print_rate=1)


Here we have included a ``print_rate`` for a progress bar, but leaving
this out will massively reduce fitting time (at the expense of not
knowing how many iterations have been run). The IMNN will be fit for a
maximum of ``max_iterations = 1000`` iterations, but with early stopping
which can turn on after ``min_iterations = 100`` iterations and after
``patience = 10`` iterations where the maximum determinant of the Fisher
information matrix has not increased. ``imnn.w`` is set to the values of
the network parameters which obtained the highest value of the
determinant of the Fisher information matrix, but the values at the
final iteration can be set using ``best = False``.

To continue training one can simply rerun fit

.. code:: python

   imnn.fit(λ, ϵ, patience=10, max_iterations=1000, print_rate=1)

although we will not run it in this example.

To visualise the fitting history we can plot the results:

.. code:: ipython3

    imnn.plot(expected_detF=50);



.. image:: output_31_0.png
