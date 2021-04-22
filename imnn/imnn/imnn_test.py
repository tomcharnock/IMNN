import pytest
import jax
import tensorflow as tf
from imnn import IMNN, SimulatorIMNN, AggregatedSimulatorIMNN, \
    GradientIMNN, AggregatedGradientIMNN, DatasetGradientIMNN, \
    NumericalGradientIMNN, AggregatedNumericalGradientIMNN, \
    DatasetNumericalGradientIMNN
from test.defaults import defaultTests
from test.simulator_defaults import simulatorTests
from test.gradient_defaults import gradientTests
from test.numerical_gradient_defaults import numericalGradientTests

defaults = defaultTests()
simulator_defaults = simulatorTests()
gradient_defaults = gradientTests()
numerical_gradient_defaults = numericalGradientTests()

host = jax.devices("cpu")[0]
devices = jax.devices()
n_devices = len(devices)
n_per_device = defaults.n_d


@pytest.mark.parametrize(
    "model",
    [defaults.model,
     tf.keras.models.Sequential([
         tf.keras.layers.Dense(128),
         tf.keras.layers.LeakyReLU(0.01),
         tf.keras.layers.Dense(128),
         tf.keras.layers.LeakyReLU(0.01),
         tf.keras.layers.Dense(128),
         tf.keras.layers.LeakyReLU(0.01),
         tf.keras.layers.Dense(defaults.n_d)])
     ])
@pytest.mark.parametrize(
    "optimiser",
    [defaults.optimiser,
     tf.keras.optimizers.Adam()
     ])
@pytest.mark.parametrize("simulator", [None, simulator_defaults.simulator])
@pytest.mark.parametrize(
    "fiducial",
    [None, gradient_defaults.fiducial, numerical_gradient_defaults.fiducial,
     [tf.data.Dataset.from_tensor_slices(
         fid).repeat().as_numpy_iterator()
      for fid in numerical_gradient_defaults.fiducial.reshape(
          (n_devices,
           defaults.n_s // (n_devices * n_per_device),
           n_per_device)
          + defaults.input_shape)]])
@pytest.mark.parametrize(
    "derivative",
    [None, gradient_defaults.derivative,
     numerical_gradient_defaults.derivative,
     [tf.data.Dataset.from_tensor_slices(
         der).repeat().as_numpy_iterator()
      for der in numerical_gradient_defaults.derivative.reshape(
          (n_devices,
           2 * defaults.n_params * defaults.n_d //
              (n_devices * n_per_device),
           n_per_device)
          + defaults.input_shape)]])
@pytest.mark.parametrize(
    "main",
    [None,
     [tf.data.Dataset.zip((
         tf.data.Dataset.from_tensor_slices(fid),
         tf.data.Dataset.from_tensor_slices(der))).repeat().as_numpy_iterator()
      for fid, der in zip(
          gradient_defaults.fiducial[:defaults.n_d].reshape(
              (n_devices,
               defaults.n_d // (n_devices * n_per_device),
               n_per_device)
              + defaults.input_shape),
          gradient_defaults.derivative.reshape(
              (n_devices,
               defaults.n_d // (n_devices * n_per_device),
               n_per_device)
              + defaults.input_shape
              + (defaults.n_params,)))]])
@pytest.mark.parametrize(
    "remaining",
    [None,
     [tf.data.Dataset.from_tensor_slices(
         fid).repeat().as_numpy_iterator()
      for fid in gradient_defaults.fiducial[defaults.n_d:].reshape(
          (n_devices,
           (defaults.n_s - defaults.n_d) // (n_devices * n_per_device),
           n_per_device)
          + defaults.input_shape)]])
@pytest.mark.parametrize(
    "δθ",
    [None, numerical_gradient_defaults.δθ])
@pytest.mark.parametrize(
    "validation_fiducial",
    [None,
     gradient_defaults.validation_fiducial,
     numerical_gradient_defaults.validation_fiducial,
     [tf.data.Dataset.from_tensor_slices(
         fid).repeat().as_numpy_iterator()
      for fid in numerical_gradient_defaults.validation_fiducial.reshape(
          (n_devices,
           defaults.n_s // (n_devices * n_per_device),
           n_per_device)
          + defaults.input_shape)]])
@pytest.mark.parametrize(
    "validation_derivative",
    [None,
     gradient_defaults.validation_derivative,
     numerical_gradient_defaults.validation_derivative,
     [tf.data.Dataset.from_tensor_slices(
         der).repeat().as_numpy_iterator()
      for der in numerical_gradient_defaults.validation_derivative.reshape(
          (n_devices,
           2 * defaults.n_params * defaults.n_d //
              (n_devices * n_per_device),
           n_per_device)
          + defaults.input_shape)]])
@pytest.mark.parametrize(
    "validation_main",
    [None,
     [tf.data.Dataset.zip((
         tf.data.Dataset.from_tensor_slices(fid),
         tf.data.Dataset.from_tensor_slices(der))).repeat().as_numpy_iterator()
      for fid, der in zip(
          gradient_defaults.validation_fiducial[:defaults.n_d].reshape(
              (n_devices,
               defaults.n_d // (n_devices * n_per_device),
               n_per_device)
              + defaults.input_shape),
          gradient_defaults.validation_derivative.reshape(
              (n_devices,
               defaults.n_d // (n_devices * n_per_device),
               n_per_device)
              + defaults.input_shape
              + (defaults.n_params,)))]])
@pytest.mark.parametrize(
    "validation_remaining",
    [None,
     [tf.data.Dataset.from_tensor_slices(
         fid).repeat().as_numpy_iterator()
      for fid in gradient_defaults.validation_fiducial[defaults.n_d:].reshape(
          (n_devices,
           (defaults.n_s - defaults.n_d) // (n_devices * n_per_device),
           n_per_device)
          + defaults.input_shape)]])
@pytest.mark.parametrize("host", [host])
@pytest.mark.parametrize("devices", [devices])
@pytest.mark.parametrize("n_per_device", [n_per_device])
@pytest.mark.parametrize("cache", [None, False])
@pytest.mark.parametrize("prefetch", [None, tf.data.AUTOTUNE])
def test_IMNN(
        model, optimiser, simulator, fiducial, derivative, main, remaining, δθ,
        validation_fiducial, validation_derivative, validation_main,
        validation_remaining, host, devices, n_per_device, cache, prefetch):
    def call_IMNN():
        return IMNN(
            n_s=defaults.n_s,
            n_d=defaults.n_d,
            n_params=defaults.n_params,
            n_summaries=defaults.n_summaries,
            input_shape=defaults.input_shape,
            θ_fid=defaults.θ_fid,
            model=model,
            optimiser=optimiser,
            key_or_state=defaults.model_key,
            simulator=simulator,
            fiducial=fiducial,
            derivative=derivative,
            main=main,
            remaining=remaining,
            δθ=δθ,
            validation_fiducial=validation_fiducial,
            validation_derivative=validation_derivative,
            validation_main=validation_main,
            validation_remaining=validation_remaining,
            host=host,
            devices=devices,
            n_per_device=n_per_device,
            cache=cache,
            prefetch=prefetch)

    numerical_shape = (defaults.n_d, 2, defaults.n_params) \
        + defaults.input_shape
    gradient_shape = (defaults.n_d,) + defaults.input_shape \
        + (defaults.n_params,)

    if (isinstance(model, tf.keras.models.Model)
            or isinstance(optimiser, tf.keras.optimizers.Optimizer)):
        with pytest.raises(ValueError):
            call_IMNN()
        return

    if ((simulator is not None)
            and (fiducial is None)
            and (derivative is None)
            and (main is None)
            and (remaining is None)
            and (δθ is None)
            and (validation_fiducial is None)
            and (validation_derivative is None)
            and (validation_main is None)
            and (validation_remaining is None)
            and (cache is None)
            and (prefetch is None)):
        if (host is None) and (devices is None) and (n_per_device is None):
            assert isinstance(call_IMNN(), SimulatorIMNN)
            return
        if ((host is not None)
                and (devices is not None)
                and (n_per_device is not None)):
            assert isinstance(call_IMNN(), AggregatedSimulatorIMNN)
            return

    if ((fiducial is not None)
            and (derivative is not None)
            and (simulator is None)
            and (((host is None)
                  and (devices is None)
                  and (n_per_device is None)
                  and (cache is None)
                  and (prefetch is None))
                 or ((host is not None)
                     and (devices is not None)
                     and (n_per_device is not None)))):
        if (any([isinstance(fiducial, list),
                 isinstance(derivative, list)])
                and (not all([isinstance(fiducial, list),
                              isinstance(derivative, list)]))):
            with pytest.raises(TypeError):
                call_IMNN()
            return
        if ((δθ is not None)
                and (((validation_fiducial is None)
                      and (validation_derivative is not None))
                     or ((validation_fiducial is not None)
                         and (validation_derivative is None)))):
            with pytest.raises(ValueError):
                call_IMNN()
            return
        if all([isinstance(fiducial, list), isinstance(derivative, list)]):
            if ((δθ is None)
                    or ((host is None)
                        and (devices is None)
                        and (n_per_device is None))):
                with pytest.raises(TypeError):
                    call_IMNN()
                return
        if (((validation_fiducial is not None)
                and (validation_derivative is not None))
                and any([isinstance(fiducial, list),
                         isinstance(derivative, list),
                         isinstance(validation_fiducial, list),
                         isinstance(validation_derivative, list)])
                and (not all([isinstance(fiducial, list),
                              isinstance(derivative, list),
                              isinstance(validation_fiducial, list),
                              isinstance(validation_derivative, list)]))):
            if ((main is not None)
                    or (remaining is not None)
                    or (validation_main is not None)
                    or (validation_remaining is not None)):
                with pytest.raises(ValueError):
                    call_IMNN()
                return
            else:
                with pytest.raises(TypeError):
                    call_IMNN()
                return

    if ((fiducial is not None)
            and (derivative is not None)
            and (main is None)
            and (remaining is None)
            and (validation_main is None)
            and (validation_remaining is None)
            and (not isinstance(model, tf.keras.models.Model))
            and (not isinstance(optimiser, tf.keras.optimizers.Optimizer))
            and (simulator is None)):
        if ((validation_fiducial is not None)
                and (validation_derivative is not None)):
            if δθ is not None:
                if (isinstance(fiducial, list)
                        and isinstance(derivative, list)
                        and isinstance(validation_fiducial, list)
                        and isinstance(validation_derivative, list)
                        and (cache is None)
                        and (prefetch is None)
                        and (host is not None)
                        and (devices is not None)
                        and (n_per_device is not None)):
                    assert isinstance(
                        call_IMNN(), DatasetNumericalGradientIMNN)
                    return
                if not any([isinstance(fiducial, list),
                            isinstance(derivative, list),
                            isinstance(validation_fiducial, list),
                            isinstance(validation_derivative, list)]):
                    if ((validation_derivative.shape == numerical_shape)
                            and (derivative.shape == numerical_shape)):
                        if ((host is None)
                                and (devices is None)
                                and (n_per_device is None)
                                and (cache is None)
                                and (prefetch is None)):
                            assert isinstance(
                                call_IMNN(), NumericalGradientIMNN)
                            return
                        if ((host is not None)
                                and (devices is not None)
                                and (n_per_device is not None)):
                            assert isinstance(
                                call_IMNN(), AggregatedNumericalGradientIMNN)
                            return
            else:
                if not any([isinstance(fiducial, list),
                            isinstance(derivative, list),
                            isinstance(validation_fiducial, list),
                            isinstance(validation_derivative, list)]):
                    if ((validation_derivative.shape == gradient_shape)
                            and (derivative.shape == gradient_shape)):
                        if ((host is None)
                                and (devices is None)
                                and (n_per_device is None)
                                and (cache is None)
                                and (prefetch is None)):
                            assert isinstance(call_IMNN(), GradientIMNN)
                            return
                        if ((host is not None)
                                and (devices is not None)
                                and (n_per_device is not None)):
                            assert isinstance(
                                call_IMNN(), AggregatedGradientIMNN)
                            return
        if ((validation_fiducial is None) and (validation_derivative is None)):
            if δθ is not None:
                if (isinstance(fiducial, list)
                        and isinstance(derivative, list)
                        and (cache is None)
                        and (prefetch is None)
                        and (host is not None)
                        and (devices is not None)
                        and (n_per_device is not None)):
                    assert isinstance(
                        call_IMNN(), DatasetNumericalGradientIMNN)
                    return
                if not any([isinstance(fiducial, list),
                            isinstance(derivative, list),
                            isinstance(validation_fiducial, list),
                            isinstance(validation_derivative, list)]):
                    if derivative.shape == numerical_shape:
                        if ((host is None)
                                and (devices is None)
                                and (n_per_device is None)
                                and (cache is None)
                                and (prefetch is None)):
                            assert isinstance(
                                call_IMNN(), NumericalGradientIMNN)
                            return
                        if ((host is not None)
                                and (devices is not None)
                                and (n_per_device is not None)):
                            assert isinstance(
                                call_IMNN(), AggregatedNumericalGradientIMNN)
                            return
            else:
                if not any([isinstance(fiducial, list),
                            isinstance(derivative, list),
                            isinstance(validation_fiducial, list),
                            isinstance(validation_derivative, list)]):
                    if derivative.shape == gradient_shape:
                        if ((host is None)
                                and (devices is None)
                                and (n_per_device is None)
                                and (cache is None)
                                and (prefetch is None)):
                            assert isinstance(
                                call_IMNN(), GradientIMNN)
                            return
                        if ((host is not None)
                                and (devices is not None)
                                and (n_per_device is not None)):
                            assert isinstance(
                                call_IMNN(), AggregatedGradientIMNN)
                            return
    if ((main is not None)
            and (remaining is not None)
            and (fiducial is None)
            and (derivative is None)
            and (validation_fiducial is None)
            and (validation_derivative is None)
            and (simulator is None)
            and (prefetch is None)
            and (cache is None)
            and (δθ is None)
            and (host is not None)
            and (devices is not None)
            and (n_per_device is not None)):
        if (((validation_main is not None)
                and (validation_remaining is not None))
                or ((validation_main is None)
                    and (validation_remaining is None))):
            assert isinstance(call_IMNN(), DatasetGradientIMNN)
            return

    with pytest.raises(ValueError):
        call_IMNN()
