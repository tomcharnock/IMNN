from test.defaults import defaultTests, simulator
import pytest
import inspect


class simulatorTests(defaultTests):
    def __init__(self, simulator=simulator, **kwargs):
        super().__init__(**kwargs)
        self.simulator = simulator
        self.kwargs["simulator"] = simulator
        self.reduced_kwargs["simulator"] = simulator
        self.simulate = True

    def specific_exceptions(self, variable, input_variable, kwargs):
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
        return False
