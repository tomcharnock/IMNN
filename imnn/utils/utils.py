import inspect
import jax
import jaxlib
import jax.numpy as np
from jax.tree_util import tree_flatten, tree_unflatten


def _check_boolean(input, name):
    """Exception raising if input is not a boolean

    Checks whether input is not ``None`` and if not checks that the input is a
    boolean.

    Parameters
    ----------
    input: any
        The input parameter to be checked
    name: str
        The name of the variable for printing explicit errors in ``Exception``

    Returns
    -------
    bool
        Returns the input if all checks pass

    Raises
    ------
    ValueError
        If input is None
    TypeError
        If input does not have correct type
    """
    if input is None:
        raise ValueError(f"`{name}` is None")
    if not ((input is True) or (input is False)):
        raise TypeError(f"`{name}` must be type {bool} but is " +
                        f"{type(input)}")
    return input


def _check_type(input, target_type, name, shape=None, allow_None=False):
    """Exception raising checks for parameter types (and shapes)

    Checks whether input is not ``None`` and if not checks that the input has
    the specified type and if not raises a warning.

    Also can check the shape of objects, although for jax numpy arrays this is
    done with :func:~`imnn.utils.check_input` since arrays can have various
    types and it is hard to predict which depending on case.

    Parameters
    ----------
    input: any
        The input parameter to be checked
    target_type: type
        The target type that the input parameter should be an instance of
    name: str
        The name of the variable for printing explicit errors in ``Exception``
    shape: tuple or int or None, default=None
        The length of a list or tuple or the shape of an array if also checked
    allow_None: bool, default=False
        Whether a ``None`` input can be returned as None without raising error

    Returns
    -------
    any
        Returns the input if all checks pass

    Raises
    ------
    ValueError
        If input is None
    ValueError
        If input shape is incorrect
    TypeError
        If input does not have correct type
    """
    if (input is None) and (not allow_None):
        raise ValueError(f"`{name}` is None")
    elif (input is None) and allow_None:
        return input
    elif not isinstance(input, target_type):
        raise TypeError(f"`{name}` must be type {target_type} but is " +
                        f"{type(input)}")
    elif shape is not None:
        if hasattr(input, "shape"):
            input_shape = input.shape
            string = "shape"
        else:
            input_shape = len(input)
            string = "length"
        if input_shape != shape:
            raise ValueError(f"`{name}` must have shape {shape} but has " +
                             f"{string} {input_shape}.")
    return input


def _check_input(input, shape, name, allow_None=False):
    """Exception raising checks for numpy array shapes

    Checks whether input is not ``None`` and if not checks that the input is a
    jax numpy array and if not raises a warning. If the input is a jax numpy
    array it then checks the shape is the same as the required shape.

    Can also allow ``None`` to be passed if it input is not essential.

    Parameters
    ----------
    input: any
        The input parameter to be checked
    shape: tuple
        The shape that the input is required to be
    name: str
        The name of the variable for printing explicit errors in ``Exception``
    allow_None: bool, default=False
        Whether a ``None`` input can be returned as None without raising error

    Returns
    -------
    array
        Returns the input if all checks pass

    Raises
    ------
    ValueError
        If input is None
    ValueError
        If input shape is incorrect
    TypeError
        If input is not a jax array
    """
    if (input is None) and (not allow_None):
        raise ValueError(f"`{name}` is None")
    elif (input is None) and allow_None:
        return input
    elif not isinstance(
            input, (jax.interpreters.xla.device_array, np.ndarray)):
        raise TypeError(f"`{name}` must be a jax array")
    else:
        if input.shape != shape:
            raise ValueError(f"`{name}` should have shape {shape} but has " +
                             f"{input.shape}")
    return input


def _check_devices(devices):
    """Exception raising checks for XLA devices

    Checks whether input is not ``None`` and if not checks that the input is a
    list of XLA devices

    Parameters
    ----------
    devices: list
        The list of XLA devices (from ``jax.devices()`` for example)

    Returns
    -------
    list
        The list of XLA devices (from ``jax.devices()`` for example)

    Raises
    ------
    ValueError
        If input is None
    ValueError
        If input shape is incorrect
    TypeError
        If inputs are not XLA devices
    """
    if (devices is None):
        raise ValueError("`devices` is None")
    elif not isinstance(devices, list):
        raise TypeError(f"`devices` must be type {list} but is " +
                        f"{type(devices)}")
    elif len(devices) < 1:
        raise ValueError("`devices` has no elements in")
    elif not all([
            isinstance(device, jaxlib.xla_extension.Device)
            for device in devices]):
        raise TypeError("all elements of `devices` must be xla devices")
    return devices


def _check_host(host):
    """Exception raising checks for XLA devices

    Checks whether input is not ``None`` and if not checks that the input is an
    XLA devices

    Parameters
    ----------
    devices: xla device
        the XLA device (from ``jax.devices()[0]`` for example)

    Returns
    -------
    xla device
        the XLA device (from ``jax.devices()[0]`` for example)

    Raises
    ------
    ValueError
        If input is None
    TypeError
        If inputs are not XLA devices
    """
    if (host is None):
        raise ValueError("`host` is None")
    elif not isinstance(host, jaxlib.xla_extension.Device):
        raise TypeError(f"`host` must be an xla device but is a {type(host)}")
    return host


def _check_model(model):
    """Exception raising for ``jax.experimental.stax``-like model

    Checks model is not ``None`` and if not checks that the input is  a tuple
    with two functions which take the correct number of inputs. For more
    information see `jax.experimental.optimizers <https://jax.readthedocs.io/e
    n/stable/jax.experimental.stax.html>`_.

    Parameters
    ----------
    model: any
        The model to be checked

    Returns
    -------
    tuple
        Returns the input if all checks pass

    Raises
    ------
    ValueError
        If input is None
    ValueError
        If optimiser is not a tuple of functions
    ValueError
        If element functions do not require the correct number of inputs
    TypeError
        If input is not a tuple of functions
    TypeError
        If elements of tuples are not functions
    """
    if model is None:
        raise ValueError("`model` is None")
    elif not isinstance(model, tuple):
        raise TypeError(f"`model` must be type {tuple} but is {type(model)}")
    else:
        if len(model) != 2:
            raise ValueError("`model` must be a tuple of two functions. The " +
                             "first for initialising the model and the " +
                             "second to call the model")
        try:
            inspect.signature(model[0]).parameters
        except Exception:
            raise TypeError("first element of `model` must take two arguments")
        if len(inspect.signature(model[0]).parameters) != 2:
            raise ValueError(
                "first element of `model` must take two arguments")
        try:
            inspect.signature(model[1]).parameters
        except Exception:
            raise TypeError(
                "second element of `model` must take three arguments")
        if len(inspect.signature(model[1]).parameters) != 3:
            raise ValueError(
                "second element of `model` must take three arguments")
    return model


def _check_model_output(output_shape, expected_shape):
    """Exception raising for shape of model output

    Parameters
    ----------
    output_shape: tuple
        Shape of the model output
    expected_shape: tuple
        Desired shape for the model output

    Raises
    ------
    ValueError
        If input does not equal the expected shape
    """
    if output_shape != expected_shape:
        raise ValueError("`model` outputs should have shape " +
                         f"{expected_shape} but is {output_shape}")


def _check_optimiser(optimiser):
    """Exception raising for ``jax.experimental.optimizers``-like optimiser

    Checks optimiser is not ``None`` and if not checks that the input is a
    ``jax.experimental.optimizers``-like instance. To allow more freedom on
    personal optimisers, if the optimiser is not an instance of a
    ``jax.experimental.optimizers`` object then it checks that the optimiser is
    at least a tuple with three functions which take the correct number of
    inputs. For more information see `jax.experimental.optimizers <https://jax.
    readthedocs.io/en/stable/jax.experimental.optimizers.html>`_.

    Parameters
    ----------
    optimiser: any
        The optimiser to be checked

    Returns
    -------
    tuple or jax.experimental.optimizers-like object
        ``jax.experimental.optimizers``-like optimiser for initialising state,
        updating parameters and state and getting parameters from state

    Raises
    ------
    ValueError
        If input is None
    ValueError
        If optimiser is not a tuple of functions or an instance of a
        ``jax.experimental.optimizers`` object
    ValueError
        If element functions do not require the correct number of inputs
    TypeError
        If input is not a tuple of functions
    TypeError
        If elements of tuples are not functions
    """
    string = "`optimiser` must be tuple of three functions. The first for " + \
        "initialising the state, the second to update the state and the " + \
        "third to get parameters from the state."
    if optimiser is None:
        raise ValueError("`optimiser` is None")
    if isinstance(optimiser, jax.experimental.optimizers.Optimizer):
        return optimiser
    else:
        try:
            length = len(optimiser)
        except Exception:
            raise TypeError(string)
        if length != 3:
            raise TypeError(string)
        try:
            inspect.signature(optimiser[0]).parameters
        except Exception:
            raise TypeError(
                "first element of `optimiser` must take one argument")
        if len(inspect.signature(optimiser[0]).parameters) != 1:
            raise ValueError(
                "first element of `optimiser` must take one argument")
        try:
            inspect.signature(optimiser[1]).parameters
        except Exception:
            raise TypeError(
                "second element of `optimiser` must take three arguments")
        if len(inspect.signature(optimiser[1]).parameters) != 3:
            raise ValueError(
                "second element of `optimiser` must take three arguments")
        try:
            inspect.signature(optimiser[2]).parameters
        except Exception:
            raise TypeError(
                "third element of `optimiser` must take one argument")
        if len(inspect.signature(optimiser[2]).parameters) != 3:
            raise ValueError(
                "third element of `optimiser` must take one argument")
        return optimiser


def _check_state(state):
    """RNG and state checking for ``jax.experimental.optimizers``-like state

    Checks state is not ``None`` and if not checks that the input is a random
    number generator or not. If the shape and type is correct for a random
    number generator then that is passed back to initialise a new state and
    network parameters. If the input is not a random number generator then the
    state is returned. Note that no specific checks are made for this state
    but expect an error in ``imnn.imnn._imnn._IMNN().fit()`` if the parameters
    cannot be obtained from the state.

    Parameters
    ----------
    state: any
        The state to be checked

    Returns
    -------
    None, int(2,) or ``jax.experimental.optimizers``-like state, None
        Returns either a random number generator or a state depending on input

    Raises
    ------
    ValueError
        If input is None
    """
    string = "`state` not a jax optimiser state - attempting to use it anyway"
    if (state is None):
        raise ValueError("`key_or_state` is None")
    elif isinstance(state, (jax.interpreters.xla.device_array, np.ndarray)):
        if state.shape == (2,):
            return None, state
        else:
            print(string)
            return state, None
    elif isinstance(state, jax.experimental.optimizers.OptimizerState):
        return state, None
    else:
        print(string)
        return state, None


def _check_simulator(simulator):
    """Checks that the simulator is callable and takes just two parameters

    For the simulator to be functional within the IMNN it must have a specific
    form where it takes a random number and a set of parameter values and
    returns a simulation generated at those parameter values.

    Parameters
    ----------
    simulator: any
        the simulator to be checked

    Returns
    -------
    fn:
        the simulator function

    Raises
    ------
    ValueError
        if simlator is None or it is a function which takes the wrong number of
        parameters
    TypeError
        if simulator is not a callable function
    """
    if simulator is None:
        raise ValueError("`simulator` is None")
    elif not callable(simulator):
        raise TypeError("`simulator` must be type <class 'function'> but is " +
                        f"{type(simulator)}")
    else:
        if len(inspect.signature(simulator).parameters) != 2:
            raise ValueError("`simulator` must take two arguments, a " +
                             "JAX prng and simulator parameters.")
        return simulator


def _check_splitting(size, name, n_devices, n_per_device):
    """Checks whether integers are exactly divisible by two other integers

    To be able to run the aggregated IMNNs, datasets have to be reshaped across
    different devices. Since ``n_s`` and ``n_d`` are known and fixed, it is
    possible to check that the reshaping can be done and return an error if the
    splitting is not possible.

    Parameters
    ----------
    size: int
        The number (of simulations) to be split over devices
    name: str
        The variable quantity name for flagging the error
    n_devices: int
        One of the numbers to be divided by (ostensibly the number of jax
        devices available to be aggregated over)
    n_per_device: int
        The other number to be divided by (the number of simulations to run in
        a batch on each device)

    Raises
    ------
    ValueError
        if the number of simulations is not exactly divisible by the number of
        devices and the number within a batch
    """
    if (size / (n_devices * n_per_device)
            != float(size // (n_devices * n_per_device))):
        raise ValueError(f"`{name}` of {size} will not split evenly between " +
                         f"{n_devices} devices when calculating " +
                         f"{n_per_device} per device.")


def _check_statistics_set(invF, dμ_dθ, invC, μ):
    """Checks that quantities needed for IMNN score compression are assigned

    Parameters
    ----------
    invF: float(n_params, n_params) or None
        The inverse of the Fisher information matrix
    dμ_dθ: float(n_summaries, n_params) or None
        The derivative of the mean of the network outputs with respect to model
        parameters
    invC: float(n_summaries, n_summaries) or None
        The inverse of the covariance of the network outputs
    μ: float(n_summaries) or None
        The mean of the network outputs

    Raises
    ------
    ValueError
        if any of the inputs are None
    """
    if (invF is None) or (dμ_dθ is None) or (invC is None) or (μ is None):
        raise ValueError(
            "Fisher information has not yet been calculated. Please run " +
            "`imnn.set_F_statistics({w}, {key}, {validate}) with " +
            "`w = imnn.final_w`, `w = imnn.best_w`, `w = imnn.inital_w` " +
            "or otherwise, `validate = True` should be set if not " +
            "simulating on the fly.")


def add_nested_pytrees(*pytrees):
    """Sums together any number of identically shaped pytrees

    All pytrees are flattened then summed and then reshaped back into their
    tree

    Parameters
    ----------
    pytrees: any number of pytrees
        the pytrees to be summed together

    Returns
    -------
    pytree:
        the single pytree resulting from summing together all input pytrees

    Todo
    ----
    No checking for identical shape is done, although it should be done really
    """
    value_flat, value_tree = tree_flatten(pytrees[0])
    for pytree in pytrees[1:]:
        next_value_flat, _ = tree_flatten(pytree)
        value_flat = list(map(sum, zip(value_flat, next_value_flat)))
    return tree_unflatten(value_tree, value_flat)
