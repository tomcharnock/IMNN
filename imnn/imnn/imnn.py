#TODO: Move TensorFlow out of function and suggest installing imnn-tf

from imnn.imnn import SimulatorIMNN, AggregatedSimulatorIMNN, GradientIMNN, \
    AggregatedGradientIMNN, NumericalGradientIMNN, \
    AggregatedNumericalGradientIMNN


def IMNN(n_s, n_d, n_params, n_summaries, input_shape, θ_fid, model, optimiser,
         key_or_state, simulator=None, fiducial=None, derivative=None, δθ=None,
         validation_fiducial=None, validation_derivative=None, devices=None,
         n_per_device=None, n_per_batch=None, cache=None, prefetch=None,
         buffer_size=None, verbose=True):

    if (simulator is None) and ((fiducial is None) or (derivative is None)):
        raise ValueError(
            "`simulator` or `fiducial` and `derivative` data necessary")

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

    if (simulator is not None) and (fiducial is not None):
        raise ValueError(
            "`simulator` and `fiducial` data provided, please only provide " +
            "one of these. If using `fiducial` ensure that `derivative` is " +
            "also provided.")

    if (simulator is not None) and (derivative is not None):
        raise ValueError(
            "`simulator` and `derivative` data provided, please only " +
            "provide one of these. If using `derivative` ensure that " +
            "`fiducial` is also provided.")

    if (simulator is not None) and (validation_fiducial is not None):
        raise ValueError(
            "`simulator` and `validation_fiducial` data provided, please " +
            "only provide one of these. If using `validation_fiducial` " +
            "ensure that `fiducial`, `derivative` and " +
            "`validation_derivative` are also provided.")

    if (simulator is not None) and (validation_derivative is not None):
        raise ValueError(
            "`simulator` and `validation_derivative` data provided, please " +
            "only provide one of these. If using `validation_derivative` " +
            "ensure that `fiducial`, `derivative` and " +
            "`validation_fiducial` are also provided.")

    if (simulator is not None) and (δθ is not None):
        raise ValueError(
            "`simulator` and `δθ` provided, please only provide one of " +
            "these. If using `δθ` for a numerical gradient ensure that " +
            "`fiducial`, `derivative` and optionally `validation_fiducial` " +
            " and `validation_derivative` are also provided.")

    if (simulator is not None) and isinstance(model, tf.keras.models.Model):
        raise ValueError(
            "`simulator` only compatible with JAX mode IMNN which is not " +
            "compatible with `keras` models. Please consider using " +
            "something like STAX or likewise for constructing the model.")

    if (simulator is not None) \
            and isinstance(optimiser, tf.keras.optimizers.Optimizer):
        raise ValueError(
            "`simulator` only compatible with JAX mode IMNN which is not " +
            "compatible with `keras` optimisers. Please consider using " +
            "something like JAX's optimisers or likewise for optimisation.")

    if (simulator is not None) and (n_per_batch is not None):
        raise ValueError(
            "`simulator` and `n_per_batch` provided. `n_per_batch` is a " +
            "TensorFlow mode parameter which is not compatible on-the-fly " +
            "simulation. If you want to aggregate the simulation " +
            "calculation, this is possible via the JAX interface by " +
            "supplying `devices=jax.devices()` and `n_per_device`.")

    if (simulator is not None) and (cache is not None):
        raise ValueError(
            "`simulator` and `cache` provided. `cache` is a TensorFlow " +
            "mode parameter which is not compatible on-the-fly simulation.")

    if (simulator is not None) and (prefetch is not None):
        raise ValueError(
            "`simulator` and `prefetch` provided. `prefetch` is a " +
            "TensorFlow mode parameter which is not compatible on-the-fly " +
            "simulation.")

    if (simulator is not None) and (buffer_size is not None):
        raise ValueError(
            "`simulator` and `buffer_size` provided. `buffer_size` is a " +
            "TensorFlow mode parameter which is not compatible on-the-fly " +
            "simulation.")

    if simulator is None:
        simulate = False
    else:
        simulate = True
        parameters["simulator"] = simulator

    if (devices is None) and (n_per_device is not None):
        raise ValueError(
            "`n_per_device` is provided, but `devices` is missing. If " +
            "aggregating computation please provide `devices=jax.devices()`.")

    if (devices is not None) and (n_per_device is None):
        raise ValueError(
            "`devices` is provided, but `n_per_device` is missing. If " +
            "aggregating computation please provide the size of parallel " +
            "calls on each device with `n_per_device`.")

    if devices is None:
        aggregate = False
    else:
        aggregate = True
        parameters["devices"] = devices
        parameters["n_per_device"] = n_per_device

    if aggregate and isinstance(model, tf.keras.models.Model):
        raise ValueError(
            "`devices` and 'n_per_device' only compatible with JAX mode " +
            "IMNN which is not compatible with `keras` models. Please " +
            "consider using something like STAX or likewise for " +
            "constructing the model.")

    if aggregate and isinstance(optimiser, tf.keras.optimizers.Optimizer):
        raise ValueError(
            "`devices` and 'n_per_device' only compatible with JAX mode " +
            "IMNN which is not compatible with `keras` optimisers. Please " +
            "consider using something like JAX's optimisers or likewise " +
            "for optimisation.")

    if aggregate and (n_per_batch is not None):
        raise ValueError(
            "`devices` and 'n_per_device' and `n_per_batch` provided. " +
            "`n_per_batch` is a TensorFlow mode parameter which is not " +
            "compatible with JAX aggregation.")

    if aggregate and (cache is not None):
        raise ValueError(
            "`devices` and 'n_per_device' and `cache` provided. " +
            "`cache` is a TensorFlow mode parameter which is not " +
            "compatible with JAX aggregation.")

    if aggregate and (prefetch is not None):
        raise ValueError(
            "`devices` and 'n_per_device' and `prefetch` provided. " +
            "`prefetch` is a TensorFlow mode parameter which is not " +
            "compatible with JAX aggregation.")

    if aggregate and (buffer_size is not None):
        raise ValueError(
            "`devices` and 'n_per_device' and `buffer_size` provided. " +
            "`buffer_size` is a TensorFlow mode parameter which is not " +
            "compatible with JAX aggregation.")

    if isinstance(model, tf.keras.models.Model) \
            and (not isinstance(optimiser, tf.keras.optimizers.Optimizer)):
        raise ValueError(
            "`model` is from `tf.keras.models.Model`, but `optimiser` is " +
            "not from `tf.keras.optimizers.Optimizer.")

    if (not isinstance(model, tf.keras.models.Model)) \
            and isinstance(optimiser, tf.keras.optimizers.Optimizer):
        raise ValueError(
            "`optimiser` is from `tf.keras.optimizers.Optimizer but `model` " +
            "is not from `tf.keras.models.Model`")

    if (not simulate) and (fiducial is None):
        raise ValueError("No `fiducial` data provided.")

    if (not simulate) and (derivative is None):
        raise ValueError("No `derivative` data provided.")

    if ((not simulate)
            and ((validation_fiducial is None)
                 and (validation_derivative is not None))):
        raise ValueError(
            "`validation_derivative` data provided but " +
            "`validation_fiducial` missing. If using validation data " +
            "(recommended) please provide both of these arguments.")

    if ((not simulate)
            and ((validation_fiducial is not None)
                 and (validation_derivative is None))):
        raise ValueError(
            "`validation_fiducial` data provided but " +
            "`validation_derivative` missing. If using validation data " +
            "(recommended) please provide both of these arguments.")

    if not simulate:
        parameters["fiducial"] = fiducial
        parameters["derivative"] = derivative
        parameters["validation_fiducial"] = validation_fiducial
        parameters["validation_derivative"] = validation_derivative

    if ((not simulate)
            and isinstance(model, tf.keras.models.Model)
            and (not isinstance(fiducial, tf.data.Dataset))):
        raise ValueError(
            "`fiducial` data must be in a tf.data.Dataset when using " +
            "TensorFlow IMNN mode.")

    if ((not simulate)
            and isinstance(model, tf.keras.models.Model)
            and (not isinstance(derivative, tf.data.Dataset))):
        raise ValueError(
            "`derivative` data must be in a tf.data.Dataset when using " +
            "TensorFlow IMNN mode.")

    if ((not simulate)
            and isinstance(model, tf.keras.models.Model)
            and (not isinstance(validation_fiducial, tf.data.Dataset))):
        raise ValueError(
            "`validation_fiducial` data must be in a tf.data.Dataset when " +
            "using TensorFlow IMNN mode.")

    if ((not simulate)
            and isinstance(model, tf.keras.models.Model)
            and (not isinstance(validation_derivative, tf.data.Dataset))):
        raise ValueError(
            "`validation_derivative` data must be in a tf.data.Dataset " +
            "when using TensorFlow IMNN mode.")

    if (not simulate) and isinstance(model, tf.keras.models.Model):
        tensorflow = True
        if n_per_batch is not None:
            parameters["n_per_batch"] = n_per_batch
        if cache is not None:
            parameters["cache"] = cache
        if prefetch is not None:
            parameters["prefetch"] = prefetch
        if buffer_size is not None:
            parameters["buffer_size"] = buffer_size
    else:
        tensorflow = False

    if tensorflow and (δθ is None):
        raise ValueError(
            "Currently TensorFlow mode IMNN is only compatible with " +
            "numercial gradients, so `δθ` must be provided.")

    if tensorflow and (not isinstance(δθ, tf.Tensor)):
        raise ValueError(
            "`δθ` is not a `tf.Tensor`, please wrap it in `tf.constant(δθ)`" +
            "when using TensorFlow mode IMNN.")

    if (not tensorflow) and isinstance(δθ, tf.Tensor):
        raise ValueError(
            "`δθ` is `tf.Tensor` which is not compatible with JAX mode IMNN.")

    if (not tensorflow) and isinstance(fiducial, tf.data.Dataset):
        raise ValueError(
            "`fiducial` data cannot be a tf.data.Dataset when using JAX " +
            "IMNN mode. Either use a keras `model` and `optimiser` or set " +
            "`fiducial` to a jax array.`")

    if (not tensorflow) and isinstance(derivative, tf.data.Dataset):
        raise ValueError(
            "`derivative` data cannot be a tf.data.Dataset when using JAX " +
            "IMNN mode. Either use a keras `model` and `optimiser` or set " +
            "`derivative` to a jax array.`")

    if validation_fiducial is not None:
        if ((not tensorflow)
                and isinstance(validation_fiducial, tf.data.Dataset)):
            raise ValueError(
                "`validation_fiducial` data cannot be a tf.data.Dataset " +
                "when using JAX IMNN mode. Either use a keras `model` and " +
                "`optimiser` or set `validation_fiducial` to a jax array.`")

    if validation_derivative is not None:
        if ((not tensorflow)
                and isinstance(validation_derivative, tf.data.Dataset)):
            raise ValueError(
                "`validation_derivative` data cannot be a tf.data.Dataset " +
                "when using JAX IMNN mode. Either use a keras `model` and " +
                "`optimiser` or set `validation_derivative` to a jax array.`")

    if (not tensorflow) and (n_per_batch is not None):
        raise ValueError(
            "`n_per_batch` is a TensorFlow mode parameter which is not " +
            "compatible with JAX mode inputs, models and optimisers.")

    if (not tensorflow) and (cache is not None):
        raise ValueError(
            "`devices` and 'n_per_device' and `cache` provided. " +
            "`cache` is a TensorFlow mode parameter which is not " +
            "compatible with JAX mode inputs, models and optimisers.")

    if (not tensorflow) and (prefetch is not None):
        raise ValueError(
            "`prefetch` is a TensorFlow mode parameter which is not " +
            "compatible with JAX mode inputs, models and optimisers.")

    if (not tensorflow) and (buffer_size is not None):
        raise ValueError(
            "`buffer_size` is a TensorFlow mode parameter which is not " +
            "compatible with JAX mode inputs, models and optimisers.")

    if tensorflow:
        if (derivative.element_spec.shape
                != tf.TensorShape((2, n_params) + input_shape)):
            raise ValueError(
                "`derivative.element_spec.shape` for numerical " +
                f"derivatives must be {(2, n_params) + input_shape} but is " +
                f"{derivative.element_spec.shape}. This is (2, n_params) + " +
                "input_shape where [:, 0] are the simulations made below " +
                "the fiducial and [:, 1] are made above the fiducial.")
        if validation_derivative is not None:
            if (validation_derivative.element_spec.shape
                    != tf.TensorShape((n_d, 2, n_params) + input_shape)):
                raise ValueError(
                    "`validation_derivative.element_spec.shape` for " +
                    "numerical derivatives must be " +
                    f"{(n_d, 2, n_params) + input_shape} but is " +
                    f"{validation_derivative.element_spec.shape}. This is " +
                    "(n_d, 2, n_params) + input_shape where [:, 0] are " +
                    "the simulations made below the fiducial and [:, 1] " +
                    "are made above the fiducial.")

    if δθ is None:
        numerical_gradient = False
    else:
        numerical_gradient = True
        parameters["δθ"] = δθ

    if numerical_gradient and (not tensorflow) and (not simulate):
        if derivative.shape != (n_d, 2, n_params) + input_shape:
            raise ValueError(
                "`derivative.shape` for numerical derivatives must be " +
                f"{(n_d, 2, n_params) + input_shape} but is " +
                f"{derivative.shape}. This is (n_d, 2, n_params) + " +
                "input_shape where [:, 0] are the simulations made below " +
                "the fiducial and [:, 1] are made above the fiducial.")
        if validation_derivative is not None:
            if validation_derivative.shape != (n_d, 2, n_params) + input_shape:
                raise ValueError(
                    "`validation_derivative.shape` for numerical " +
                    "derivatives must be " +
                    f"{(n_d, 2, n_params) + input_shape} but is " +
                    f"{validation_derivative.shape}. This is " +
                    "(n_d, 2, n_params) + input_shape where [:, 0] are " +
                    "the simulations made below the fiducial and [:, 1] " +
                    "are made above the fiducial.")

    if (not simulate) and (not numerical_gradient):
        if derivative.shape != (n_d,) + input_shape + (n_params,):
            raise ValueError(
                f"`derivative.shape` must be {(n_d,) + input_shape + (2,)} " +
                "when not doing numerical gradients, but is " +
                f"{derivative.shape}.")

        if validation_derivative is not None:
            if (validation_derivative.shape
                    != (n_d,) + input_shape + (n_params,)):
                raise ValueError(
                    "`validation_derivative.shape` must be " +
                    "{(n_d,) + input_shape + (2,)} when not doing " +
                    "numerical gradients but is " +
                    f"{validation_derivative.shape}.")

    if simulate:
        if aggregate:
            if verbose:
                print("`simulator` provided, using AggregatedSimulatorIMNN")
            return AggregatedSimulatorIMNN(**parameters)
        else:
            if verbose:
                print("`simulator` provided, using SimulatorIMNN")
            return SimulatorIMNN(**parameters)

    if numerical_gradient:
        if aggregate:
            if verbose:
                print("`δθ` provided, using AggregatedNumericalGradientIMNN")
            return AggregatedNumericalGradientIMNN(**parameters)
        elif tensorflow:
            if verbose:
                print("`δθ` provided, using TensorFlowNumericalGradientIMNN")
            return TensorFlowNumericalGradientIMNN(**parameters)
        else:
            if verbose:
                print("`δθ` provided, using NumericalGradientIMNN")
            return NumericalGradientIMNN(**parameters)

    if aggregate:
        if verbose:
            print("using AggregatedGradientIMNN")
        return AggregatedGradientIMNN(**parameters)
    if verbose:
        print("using GradientIMNN")
    return GradientIMNN(**parameters)
