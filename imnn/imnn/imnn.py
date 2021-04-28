from imnn.imnn import SimulatorIMNN, AggregatedSimulatorIMNN, GradientIMNN, \
    AggregatedGradientIMNN, DatasetGradientIMNN, NumericalGradientIMNN, \
    AggregatedNumericalGradientIMNN, DatasetNumericalGradientIMNN
import tensorflow as tf


def IMNN(n_s, n_d, n_params, n_summaries, input_shape, θ_fid, model, optimiser,
         key_or_state, simulator=None, fiducial=None, derivative=None,
         main=None, remaining=None, δθ=None, validation_fiducial=None,
         validation_derivative=None, validation_main=None,
         validation_remaining=None, host=None, devices=None, n_per_device=None,
         cache=None, prefetch=None, verbose=True):
    """Selection function to return correct submodule based on inputs

    Because there are many different subclasses to work with specific types of
    simulations (or a simulator) and how their gradients are calculated this
    function provides a way to try and return the desired one based on the data
    passed.

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
        ``fn(w: list, d: float([None], input_shape)) -> float([None],
        n_summaries)``.
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
    simulator : fn, optional requirement
        (:func:`~immn.SimulatorIMNN`, :func:`~immn.AggregatedSimulatorIMNN`)
        A function that generates a single simulation from a random number
        generator and a tuple (or array) of parameter values at which to
        generate the simulations. For the purposes of use in LFI/ABC afterwards
        it is also useful for the simulator to be able to broadcast to a batch
        of simulations on the zeroth axis
        ``fn(int(2,), float([None], n_params)) -> float([None], input_shape)``
    fiducial : float or list, optional requirement
        The simulations generated at the fiducial model parameter values
        used for calculating the covariance of network outputs
        (for fitting)

            - *(float(n_s, input_shape))* -- :func:`~immn.GradientIMNN`,
              :func:`~immn.NumericalGradientIMNN`,
              :func:`~immn.AggregatedGradientIMNN`,
              :func:`~immn.AggregatedNumericalGradientIMNN`
            - *(list of numpy iterators)* --
              :func:`~immn.DatasetNumericalGradientIMNN`

    derivative : float or list, optional requirement
        The simulations generated at parameter values perturbed from the
        fiducial used to calculate the numerical derivative of network
        outputs with respect to model parameters (for fitting)

            - *(float(n_d, input_shape, n_params))* --
              :func:`~immn.GradientIMNN`,
              :func:`~immn.AggregatedGradientIMNN`
            - *(float(n_d, 2, n_params, input_shape))* --
              :func:`~immn.NumericalGradientIMNN`,
              :func:`~immn.AggregatedNumericalGradientIMNN`
            - *(list of numpy iterators)* --
              :func:`~immn.DatasetNumericalGradientIMNN`

    main : list of numpy iterators, optional requirement
        (:func:`~immn.DatasetGradientIMNN`) The simulations generated at the
        fiducial model parameter values used for calculating the covariance of
        network outputs and their derivatives with respect to the physical
        model parameters (for fitting). These are served ``n_per_device`` at a
        time as a numpy iterator from a TensorFlow dataset.
    remaining : list of numpy iterators, optional requirement
        (:func:`~immn.DatasetGradientIMNN`) The ``n_s - n_d`` simulations
        generated at the fiducial model parameter values used for calculating
        the covariance ofnetwork outputs with a derivative counterpart (for
        fitting). These are served ``n_per_device`` at a time as a numpy
        iterator from a TensorFlow dataset.
    δθ : float(n_params,), optional requirement
        (:func:`~immn.NumericalGradientIMNN`,
        :func:`~immn.AggregatedNumericalGradientIMNN`,
        :func:`~immn.DatasetNumericalGradientIMNN`) Size of perturbation to
        model parameters for the numerical derivative
    validation_fiducial : float or list, optional requirement
        The simulations generated at the fiducial model parameter values
        used for calculating the covariance of network outputs
        (for validation)

            - *(float(n_s, input_shape))* --
              :func:`~immn.GradientIMNN`,
              :func:`~immn.NumericalGradientIMNN`,
              :func:`~immn.AggregatedGradientIMNN`,
              :func:`~immn.AggregatedNumericalGradientIMNN`
            - *(list of numpy iterators)* --
              :func:`~immn.DatasetNumericalGradientIMNN`

    validation_derivative : float or list, optional requirement
        The simulations generated at parameter values perturbed from the
        fiducial used to calculate the numerical derivative of network
        outputs with respect to model parameters (for validation)

            - *(float(n_d, input_shape, n_params))* --
              :func:`~immn.GradientIMNN`,
              :func:`~immn.AggregatedGradientIMNN`
            - *(float(n_d, 2, n_params, input_shape))* --
              :func:`~immn.NumericalGradientIMNN`,
              :func:`~immn.AggregatedNumericalGradientIMNN`
            - *(list of numpy iterators)* --
              :func:`~immn.DatasetNumericalGradientIMNN`

    validation_main : list of numpy iterators, optional requirement
        (:func:`~immn.DatasetGradientIMNN`) The simulations generated at the
        fiducial model parameter values used for calculating the covariance of
        network outputs and their derivatives with respect to the physical
        model parameters (for validation). These are served ``n_per_device`` at
        a time as a numpy iterator from a TensorFlow dataset.
    validation_remaining : list of numpy iterators, optional requirement
        (:func:`~immn.DatasetGradientIMNN`) The ``n_s - n_d`` simulations
        generated at the fiducial model parameter values used for calculating
        the covariance of network outputs with a derivative counterpart (for
        validation). These are served ``n_per_device`` at a time as a numpy
        iterator from a TensorFlow dataset.
    host: jax.device, optional requirement
        (:func:`~immn.AggregatedSimulatorIMNN`,
        :func:`~immn.AggregatedGradientIMNN`,
        :func:`~immn.AggregatedNumericalGradientIMNN`,
        :func:`~immn.DatasetGradientIMNN`,
        :func:`~immn.DatasetNumericalGradientIMNN`) The main device where the
        Fisher calculation is performed (something like
        ``jax.devices("cpu")[0]``)
    devices: list, optional requirement
        (:func:`~immn.AggregatedSimulatorIMNN`,
        :func:`~immn.AggregatedGradientIMNN`,
        :func:`~immn.AggregatedNumericalGradientIMNN`,
        :func:`~imnn.DatasetGradientIMNN`,
        :func:`~immn.DatasetNumericalGradientIMNN`) A list of the available jax
        devices (from ``jax.devices()``)
    n_per_device: int, optional requirement
        (:func:`~immn.AggregatedSimulatorIMNN`,
        :func:`~immn.AggregatedGradientIMNN`,
        :func:`~immn.AggregatedNumericalGradientIMNN`,
        :func:`~immn.DatasetGradientIMNN`,
        :func:`~immn.DatasetNumericalGradientIMNN`) Number of simulations to
        handle at once, this should be as large as possible without letting the
        memory overflow for the best performance
    prefetch : tf.data.AUTOTUNE or int or None, optional, default=None
        How many simulation to prefetch in the tensorflow dataset (could be
        used in :func:`~immn.AggregatedGradientIMNN` and
        :func:`~immn.AggregatedNumericalGradientIMNN`)
    cache : bool, optional, default=None
        Whether to cache simulations in the tensorflow datasets (could be used
        in :func:`~immn.AggregatedGradientIMNN` and
        :func:`~immn.AggregatedNumericalGradientIMNN`)
    """

    if isinstance(model, tf.keras.models.Model):
        raise ValueError(
            "`model` cannot be an instance of `tf.keras.models.Model` when " +
            "using JAX mode IMNN. This is now implemented in `imnn_tf` " +
            "which is a legacy version of the imnn using the final version " +
            "the TensorFlow implementation (v0.2.0). Please consider " +
            "installing this via pip: `pip install imnn_tf`")

    if isinstance(optimiser, tf.keras.optimizers.Optimizer):
        raise ValueError(
            "`optimiser` cannot be an instance of " +
            "`tf.keras.optimizers.Optimizer` when using JAX mode IMNN. This " +
            "is now implemented in `imnn_tf` which is a legacy version of " +
            "the imnn using the final version the TensorFlow implementation " +
            "(v0.2.0). Please consider installing this via pip: " +
            "`pip install imnn_tf`")

    aggregate = False

    parameters = {
        "n_s": n_s,
        "n_d": n_d,
        "n_params": n_params,
        "n_summaries": n_summaries,
        "input_shape": input_shape,
        "θ_fid": θ_fid,
        "model": model,
        "optimiser": optimiser,
        "key_or_state": key_or_state}

    if host is None:
        if devices is None:
            if n_per_device is None:
                if cache is None:
                    if prefetch is not None:
                        raise ValueError(
                            "`prefetch` is provided although no other " +
                            "aggregation parameters are passed")
                else:
                    if prefetch is None:
                        raise ValueError(
                            "`cache` is provided although no other " +
                            "aggregation parameters are passed")
                    else:
                        raise ValueError(
                            "`cache` and `prefetch` are provided although " +
                            "no other aggregation parameters are passed")
            else:
                raise ValueError(
                    "`n_per_device` is provided, but `host` and `devices` " +
                    "are missing. If aggregating computation please provide " +
                    "something like `host=jax.devices('cpu')[0]` and " +
                    "`devices=jax.devices()`.")
        else:
            if n_per_device is None:
                raise ValueError(
                    "`devices` is provided, but `host` and `n_per_device` " +
                    "are missing. If aggregating computation please " +
                    "provide something like `host=jax.devices('cpu')[0]` " +
                    "and the number of parallel calls on each device with " +
                    "`n_per_device`.")
            else:
                raise ValueError(
                    "`devices` and `n_per_device` are provided, but `host` " +
                    "is missing. If aggregating computation please provide " +
                    "something like `host=jax.devices('cpu')[0]`")
    else:
        if devices is None:
            if n_per_device is None:
                raise ValueError(
                    "`host` is provided, but `devices` and `n_per_device` " +
                    "are missing. If aggregating computation please provide " +
                    "something like `devices=jax.devices()` and the " +
                    "number of parallel calls on each device with " +
                    "`n_per_device`.")
            else:
                raise ValueError(
                    "`host` and `n_per_device` are provided, but `devices` " +
                    "is missing. If aggregating computation please provide " +
                    "something like `devices=jax.devices()`.")
        else:
            if n_per_device is None:
                raise ValueError(
                    "`host` and `devices` are provided, but `n_per_device` " +
                    "is missing. If aggregating computation please provide " +
                    "the number of parallel calls on each device with " +
                    "`n_per_device`.")
            aggregate = True
            parameters["host"] = host
            parameters["devices"] = devices
            parameters["n_per_device"] = n_per_device

    if simulator is None:
        if fiducial is None:
            if derivative is None:
                if main is None:
                    if remaining is None:
                        raise ValueError(
                            "`simulator` or `fiducial` and `derivative` or " +
                            "`main` and `remaining` argments are necessary")
                    else:
                        raise ValueError(
                            "`remaining` simulations provided, but `main` " +
                            "simulations (with simulations and accompanying " +
                            "derivatives with respect to model parameters) " +
                            "missing.")
                else:
                    if remaining is None:
                        raise ValueError(
                            "`main` simulations provided, but `remaining` " +
                            "simulations are missing. Even if `n_s = n_d` " +
                            "`remaining` needs to be supplied (as a list of " +
                            " numpy iterators with no iterations).")
                    else:
                        if not aggregate:
                            raise ValueError(
                                "aggregation parameters (`host`, `devices` " +
                                "and `n_per_device`) are required when " +
                                "passing `main` and `remaining`")
                        if δθ is not None:
                            raise ValueError(
                                "`δθ` should not be passed when providing " +
                                "`main` and `remaining`. `δθ` is used when " +
                                "calculating numerical gradients, whereas " +
                                "`main` should contain simulations and " +
                                "actual gradients")
                        if cache is not None:
                            raise ValueError(
                                "`cache` should not be passed when passing " +
                                "`main` and `remaining` since the cache " +
                                "should already be built into the dataset " +
                                "if desired")
                        if prefetch is not None:
                            raise ValueError(
                                "`prefetch` should not be passed when " +
                                "passing `main` and `remaining` since " +
                                "prefetching should already be built into " +
                                "the dataset if desired")
                        if validation_fiducial is not None:
                            raise ValueError(
                                "`main` and `remaining` simulations are " +
                                "provided, but `validation_fiducial` is " +
                                "passed too. Validation can only be done " +
                                "using `validation_main` and " +
                                "`validation_remaining` when `main` and " +
                                "`remaining` are supplied.")
                        if validation_derivative is not None:
                            raise ValueError(
                                "`main` and `remaining` simulations are " +
                                "provided, but `validation_derivative` is " +
                                "passed too. Validation can only be done " +
                                "using `validation_main` and " +
                                "`validation_remaining` when `main` and " +
                                "`remaining` are supplied.")
                        if validation_main is not None:
                            if validation_remaining is None:
                                raise ValueError(
                                    "`validation_main` is provided, but " +
                                    "`validation_remaining is missing`")
                            else:
                                parameters["validation_main"] = validation_main
                                parameters["validation_remaining"] = \
                                    validation_remaining
                        else:
                            if validation_remaining is not None:
                                raise ValueError(
                                    "`validation_remaining` is provided, " +
                                    "but `validation_main` is missing")
                        parameters["main"] = main
                        parameters["remaining"] = remaining
                        if verbose:
                            print("`main` and `required` provided, using " +
                                  "`DatasetGradientIMNN`")
                        return DatasetGradientIMNN(**parameters)
            else:
                raise ValueError(
                    "`derivative` supplied but `fiducial` is missing")
        else:
            if derivative is None:
                raise ValueError(
                    "`fiducial` supplied but `derivative` is missing")
            elif (isinstance(fiducial, list)
                    and (not isinstance(derivative, list))):
                raise TypeError("`fiducial` is a list but `derivative` is not")
            elif ((not isinstance(fiducial, list))
                    and isinstance(derivative, list)):
                raise TypeError("`derivative` is a list but `fiducial` is not")
            elif (isinstance(fiducial, list) and (δθ is None)):
                raise TypeError(
                    "`fiducial` and `derivative` should not be lists when " +
                    "passing not `δθ`. If numerical derivatives are " +
                    "intended the please provide `δθ`, and if not, but you " +
                    "want to use a list of dataset iterations then please " +
                    "construct `main` and `remaining`.")
            else:
                if main is not None:
                    if remaining is None:
                        raise ValueError(
                            "`fiducial`, `derivative` and `main` are " +
                            "supplied but only either `main` and " +
                            "`remaining` OR `fiducial` and `derivative` " +
                            "should be passed")
                    else:
                        raise ValueError(
                            "`fiducial` and `derivative` AND `main` and " +
                            "`remaining` are supplied but only either " +
                            "`main` and `remaining` OR `fiducial` and " +
                            "`derivative` should be passed")
                else:
                    if remaining is not None:
                        raise ValueError(
                            "`fiducial`, `derivative` and `remaining` are " +
                            "supplied but only either `main` and " +
                            "`remaining` OR `fiducial` and `derivative` " +
                            "should be passed")
                    if validation_main is not None:
                        raise ValueError(
                            "`fiducial` and `derivative` simulations are " +
                            "provided, but `validation_main` is " +
                            "passed too. Validation can only be done " +
                            "using `validation_fiducial` and " +
                            "`validation_derivative` when `fiducial` and " +
                            "`derivative` are supplied.")
                    if validation_remaining is not None:
                        raise ValueError(
                            "`fiducial` and `derivative` simulations are " +
                            "provided, but `validation_remaining` is " +
                            "passed too. Validation can only be done " +
                            "using `validation_fiducial` and " +
                            "`validation_derivative` when `fiducial` and " +
                            "`derivative` are supplied.")
                    if validation_fiducial is not None:
                        if validation_derivative is None:
                            raise ValueError(
                                "`validation_fiducial` is provided, but " +
                                "`validation_derivative` is missing")
                        elif (isinstance(validation_fiducial, list)
                                and (not isinstance(
                                    validation_derivative, list))):
                            raise TypeError(
                                "`validation_fiducial` is a list but " +
                                "`validation_derivative` is not")
                        elif ((not isinstance(validation_fiducial, list))
                                and isinstance(validation_derivative, list)):
                            raise TypeError(
                                "`validation_derivative` is a list but " +
                                "`validation_fiducial` is not")
                        elif (isinstance(fiducial, list)
                                and (not isinstance(
                                    validation_fiducial, list))):
                            raise TypeError(
                                "`fiducial` and `derivative` are lists but " +
                                "`validation_fiducial` and " +
                                "`validation_derivative` are not")
                        elif ((not isinstance(fiducial, list))
                                and isinstance(validation_fiducial, list)):
                            raise TypeError(
                                "`fiducial` and `derivative` are not lists " +
                                " but `validation_fiducial` and " +
                                "`validation_derivative` are")
                        else:
                            parameters["validation_fiducial"] = \
                                validation_fiducial
                            parameters["validation_derivative"] = \
                                validation_derivative
                    else:
                        if validation_derivative is not None:
                            raise ValueError(
                                "`validation_derivative` is provided, but " +
                                "`validation_fiducial` is missing")
                    parameters["fiducial"] = fiducial
                    parameters["derivative"] = derivative
    else:
        if fiducial is not None:
            raise ValueError(
                "`simulator` and `fiducial` provided, please only provide " +
                "one of these. If using `fiducial` ensure that `derivative` " +
                "is also provided.")
        if derivative is not None:
            raise ValueError(
                "`simulator` and `derivative` provided, please only provide " +
                "one of these. If using `derivative` ensure that `fiducial` " +
                "is also provided.")
        if main is not None:
            raise ValueError(
                "`simulator` and `main` provided, please only provide " +
                "one of these. If using `main` ensure that `remaining` " +
                "is also provided.")
        if remaining is not None:
            raise ValueError(
                "`simulator` and `remaining` provided, please only provide " +
                "one of these. If using `remaining` ensure that `main` " +
                "is also provided.")
        if validation_fiducial is not None:
            raise ValueError(
                "`simulator` and `validation_fiducial` provided, please " +
                "only provide one of these. If using `validation_fiducial` " +
                "ensure that `fiducial`, `derivative` and " +
                "`validation_derivative` are also provided.")
        if validation_derivative is not None:
            raise ValueError(
                "`simulator` and `validation_derivative` provided, please " +
                "only provide one of these. If using " +
                "`validation_derivative`ensure that `fiducial`, " +
                "`derivative` and `validation_fiducial` are also provided.")
        if validation_main is not None:
            raise ValueError(
                "`simulator` and `validation_main` provided, please only " +
                "provide one of these. If using `validation_main` ensure " +
                "that `main`, `remaining` and `validation_remaining` are " +
                "also provided.")
        if validation_remaining is not None:
            raise ValueError(
                "`simulator` and `validation_remaining` provided, please " +
                "only provide one of these. If using `validation_remaining` " +
                "ensure that `main`, `remaining` and `validation_main` are " +
                "also provided.")
        if δθ is not None:
            raise ValueError(
                "`δθ` should not be passed when providing `simulator`. `δθ` " +
                "is used when calculating numerical gradients, whereas the " +
                "automatic derivatives are calculated using the simulator")
        if cache is not None:
            raise ValueError(
                "`cache` should not be passed when passing `simulator` " +
                "since TensorFlow datasets are not used during simulation")
        if prefetch is not None:
            raise ValueError(
                "`prefetch` should not be passed when passing `simulator` " +
                "since TensorFlow datasets are not used during simulation")
        parameters["simulator"] = simulator
        if aggregate:
            if verbose:
                print("`simulator` provided, using AggregatedSimulatorIMNN")
            return AggregatedSimulatorIMNN(**parameters)
        else:
            if verbose:
                print("`simulator` provided, using SimulatorIMNN")
            return SimulatorIMNN(**parameters)

    if cache is not None:
        parameters["cache"] = cache
    if prefetch is not None:
        parameters["prefetch"] = prefetch

    if δθ is None:
        try:
            if derivative.shape != (n_d,) + input_shape + (n_params,):
                raise ValueError(
                    "`derivative.shape` must be " +
                    f"{(n_d,) + input_shape + (2,)} when not doing " +
                    f"numerical gradients, but is {derivative.shape}. If " +
                    "numerical gradients are intended, perhaps `δθ` has not " +
                    "been provided.")
        except AttributeError:
            raise TypeError(
                "`derivative` does not have a `shape` attribute suggesting " +
                "that the wrong type is being passed. `derivative` must be " +
                f"a numpy (jax) array but has type {type(derivative)}")
        if validation_derivative is not None:
            try:
                if validation_derivative.shape != \
                        (n_d,) + input_shape + (n_params,):
                    raise ValueError(
                        "`validation_derivative.shape` must be " +
                        f"{(n_d,) + input_shape + (2,)} when not doing " +
                        "numerical gradients but is " +
                        f"{validation_derivative.shape}. If numerical " +
                        "gradients are intended, perhaps `δθ` has not been " +
                        "provided.")
            except AttributeError:
                raise TypeError(
                    "`validation_derivative does not have a `shape` " +
                    "attribute suggesting that the wrong type is being " +
                    "passed. `validation_derivative` must be a numpy (jax) " +
                    f"array but has type {type(validation_derivative)}")
        if aggregate:
            if verbose:
                print("using AggregatedGradientIMNN")
            return AggregatedGradientIMNN(**parameters)
        else:
            if verbose:
                print("using GradientIMNN")
            return GradientIMNN(**parameters)
    else:
        parameters["δθ"] = δθ
        if isinstance(fiducial, list):
            if not aggregate:
                raise TypeError(
                    "`fiducial` and `derivative` are lists suggesting " +
                    "that they contain numpy iterators over datasets to " +
                    "be used with `DatasetNumericalGradientIMNN`, but " +
                    "no aggregation parameters passed")
            if cache is not None:
                raise ValueError(
                    "`cache` should not be passed when passing `fiducial` " +
                    "and `derivative` as lists since the cache should " +
                    "already be built into the dataset if desired")
            if prefetch is not None:
                raise ValueError(
                    "`prefetch` should not be passed when passing " +
                    "`fiducial` and `derivative` as lists since the " +
                    "prefetching should already be built into the dataset " +
                    "if desired")
            if verbose:
                print("using DatasetNumericalGradientIMNN")
            return DatasetNumericalGradientIMNN(**parameters)
        else:
            try:
                if derivative.shape != (n_d, 2, n_params) + input_shape:
                    raise ValueError(
                        "`derivative.shape` for numerical derivatives must " +
                        f"be {(n_d, 2, n_params) + input_shape} but is " +
                        f"{derivative.shape}. This is (n_d, 2, n_params) + " +
                        "input_shape where [:, 0] are the simulations made " +
                        "below the fiducial and [:, 1] are made above the " +
                        "fiducial.  If numerical gradients are not intended " +
                        "perhaps `δθ` has accidentally been provided.")
            except AttributeError:
                raise TypeError(
                    "`derivative` does not have a `shape` attribute " +
                    "suggesting that the wrong type is being passed. " +
                    "`derivative` must be a numpy (jax) array but has type " +
                    f"{type(derivative)}")
            if validation_derivative is not None:
                try:
                    if validation_derivative.shape != \
                            (n_d, 2, n_params) + input_shape:
                        raise ValueError(
                            "`validation_derivative.shape` for numerical " +
                            "derivatives must be " +
                            f"{(n_d, 2, n_params) + input_shape} but is " +
                            f"{validation_derivative.shape}. This is " +
                            "(n_d, 2, n_params) + input_shape where [:, 0] " +
                            "are the simulations made below the fiducial " +
                            "and [:, 1] are made above the fiducial. If " +
                            "numerical gradients are not intended perhaps " +
                            "`δθ` has accidentally been provided.")
                except AttributeError:
                    raise TypeError(
                        "`validation_derivative does not have a `shape` " +
                        "attribute suggesting that the wrong type is being " +
                        "passed. `validation_derivative` must be a numpy " +
                        "(jax) array but has type " +
                        f"{type(validation_derivative)}")
            if aggregate:
                if verbose:
                    print("using AggregatedNumericalGradientIMNN")
                return AggregatedNumericalGradientIMNN(**parameters)
            else:
                if verbose:
                    print("using NumericalGradientIMNN")
                return NumericalGradientIMNN(**parameters)

    print(
        "I would have thought all options should have been exhausted. " +
        "Please check that all parameters are correct - if they definitely " +
        "are then there is an error in the logic of this code, but you " +
        "probably also know what you are doing enough to directly use the " +
        "specific subclass.")
