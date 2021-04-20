import pytest
import jax.numpy as np
import jaxlib
import inspect
from test.aggregated_defaults import aggregatedTests
from test.simulator_defaults import simulatorTests
from imnn.imnn import AggregatedSimulatorIMNN


class aggregatedSimulatorTests(aggregatedTests, simulatorTests):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def specific_exceptions(self, variable, input_variable, kwargs):
        if variable == "devices":
            if input_variable is list():
                with pytest.raises(TypeError) as info:
                    self.imnn(**kwargs)
                assert info.match("`devices` has no elements in")
                return True
        if variable == "simulator":
            if callable(input_variable):
                if len(inspect.signature(input_variable).parameters) != 2:
                    kwargs["simulator"] = input_variable
                    with pytest.raises(ValueError) as info:
                        self.imnn(**kwargs)
                    assert info.match(
                        "`simulator` must take two arguments, a JAX prng " +
                        "and simulator parameters.")
                    return True
        if variable == "host":
            if input_variable is None:
                return False
            if not isinstance(input_variable, jaxlib.xla_extension.Device):
                kwargs[variable] = input_variable
                with pytest.raises(TypeError) as info:
                    self.imnn(**kwargs)
                assert info.match(
                    "`host` must be an xla device but is a "
                    f"{type(input_variable)}")
                return True
        return False


test = aggregatedSimulatorTests(
    imnn=AggregatedSimulatorIMNN, filename="aggregated_simulator",
    simulate=True)


def fail_fn(a, b, c):
    pass


@pytest.mark.parametrize("kwargs", [test.kwargs, test.reduced_kwargs])
@pytest.mark.parametrize("state", [True, False])
@pytest.mark.parametrize(
    "input_variable",
    [None, list(), 1., 1, np.zeros((1,)), test.rng, tuple(), (0, 0),
     (test.model[0], 0), test.bad_model, test.state, fail_fn])
@pytest.mark.parametrize("variable", test.kwargs.keys())
def test_initialisation_parameters_(
        variable, kwargs, input_variable, state):
    test.initialise_parameters(
        variable, kwargs, input_variable, state=state, validate=False)


@pytest.mark.parametrize("state", [False, True])
@pytest.mark.parametrize("variable", ["n_s", "n_d", "same"])
def test_splitting_(variable, state):
    test.splitting(variable, test.kwargs, state=state, validate=False)


@pytest.mark.parametrize("kwargs", [test.kwargs, test.reduced_kwargs])
@pytest.mark.parametrize("state", [True, False])
@pytest.mark.parametrize(
    "input_variable", [None, list(), 1., 1, np.zeros((1,)), test.rng])
@pytest.mark.parametrize("variable", test.fit_kwargs.keys())
def test_fit_parameters_(variable, kwargs, input_variable, state):
    test.fit_parameters(
        variable, kwargs, test.fit_kwargs, input_variable, state=state,
        validate=False)


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
