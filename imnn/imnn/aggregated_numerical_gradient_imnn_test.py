import pytest
import jax.numpy as np
import tensorflow as tf
from test.aggregated_defaults import aggregatedTests
from test.numerical_gradient_defaults import numericalGradientTests
from imnn.imnn import AggregatedNumericalGradientIMNN


class aggregatedNumericalGradientTests(
        aggregatedTests, numericalGradientTests):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kwargs["prefetch"] = tf.data.AUTOTUNE
        self.reduced_kwargs["prefetch"] = tf.data.AUTOTUNE
        self.kwargs["cache"] = True
        self.reduced_kwargs["cache"] = True


test = aggregatedNumericalGradientTests(
    imnn=AggregatedNumericalGradientIMNN,
    filename="aggregated_numerical_gradient")


@pytest.mark.parametrize("kwargs", [test.kwargs, test.reduced_kwargs])
@pytest.mark.parametrize("state", [True, False])
@pytest.mark.parametrize("validate", [True, False])
@pytest.mark.parametrize(
    "input_variable",
    [None, list(), 1., 1, np.zeros((1,)), test.rng, tuple(), (0, 0),
     (test.model[0], 0), test.bad_model, test.state])
@pytest.mark.parametrize("variable", test.kwargs.keys())
def test_initialisation_parameters_(
        variable, kwargs, input_variable, state, validate):
    test.initialise_parameters(
        variable, kwargs, input_variable, state=state, validate=validate)


@pytest.mark.parametrize("validate", [True, False])
@pytest.mark.parametrize("state", [False, True])
@pytest.mark.parametrize("variable", ["n_s", "n_d", "same"])
def test_splitting_(variable, validate, state):
    test.splitting(variable, test.kwargs, state=state, validate=validate)


@pytest.mark.parametrize("kwargs", [test.kwargs, test.reduced_kwargs])
@pytest.mark.parametrize("state", [True, False])
@pytest.mark.parametrize("validate", [True, False])
@pytest.mark.parametrize(
    "input_variable", [None, list(), 1., 1, np.zeros((1,)), test.rng])
@pytest.mark.parametrize("variable", test.fit_kwargs.keys())
def test_fit_parameters_(variable, kwargs, input_variable, state, validate):
    test.fit_parameters(
        variable, kwargs, test.fit_kwargs, input_variable, state=state,
        validate=validate)


@pytest.mark.parametrize("state", [True, False])
@pytest.mark.parametrize("validate", [True, False])
@pytest.mark.parametrize("fit", [True, False])
@pytest.mark.parametrize("none_first", [True, False])
@pytest.mark.parametrize("kwargs", [test.kwargs, test.reduced_kwargs])
def test_combined_running_test_(kwargs, state, validate, fit, none_first):
    test.combined_running_test(
        [test.single_target_data, test.batch_target_data], kwargs,
        test.fit_kwargs, state=state, validate=validate, fit=fit,
        none_first=none_first, aggregated=True)
