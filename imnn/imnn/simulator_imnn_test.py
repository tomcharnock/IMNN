import pytest
import jax.numpy as np
from test.simulator_defaults import simulatorTests
from imnn.imnn import SimulatorIMNN

test = simulatorTests(imnn=SimulatorIMNN, filename="simulator")


def fail_fn(a, b, c):
    pass


# Test that all initialisation parameters correctly raise errors
@pytest.mark.parametrize("kwargs", [test.kwargs, test.reduced_kwargs])
@pytest.mark.parametrize("state", [True, False])
@pytest.mark.parametrize(
    "input_variable",
    [None, list(), 1., 1, np.zeros((1,)), test.rng, tuple(), (0, 0),
     (test.model[0], 0), test.bad_model, test.state, fail_fn])
@pytest.mark.parametrize("variable", test.kwargs.keys())
def test_initialisation_parameters_(variable, kwargs, input_variable, state):
    test.initialise_parameters(
        variable, kwargs, input_variable, state=state, validate=False)


# Test that all parameters passed to fit correctly raise errors
@pytest.mark.parametrize("kwargs", [test.kwargs, test.reduced_kwargs])
@pytest.mark.parametrize("state", [True, False])
@pytest.mark.parametrize(
    "input_variable", [None, list(), 1., 1, np.zeros((1,)), test.rng])
@pytest.mark.parametrize("variable", test.fit_kwargs.keys())
def test_fit_parameters_(variable, kwargs, input_variable, state):
    test.fit_parameters(
        variable, kwargs, test.fit_kwargs, input_variable, state=state,
        validate=False)


# Test that fitting correctly fails and get_estimate won't run and that plot
# can be made
@pytest.mark.parametrize("state", [True, False])
@pytest.mark.parametrize("validate", [True, False])
@pytest.mark.parametrize("fit", [True, False])
@pytest.mark.parametrize("none_first", [True, False])
@pytest.mark.parametrize("kwargs", [test.kwargs, test.reduced_kwargs])
def test_combined_running_test_(kwargs, state, validate, fit, none_first):
    test.combined_running_test(
        [test.single_target_data, test.batch_target_data], kwargs,
        test.fit_kwargs, state=state, validate=validate, fit=fit,
        none_first=none_first)
