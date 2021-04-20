from test.defaults import defaultTests
import pytest
import jax
import jaxlib
import copy


class aggregatedTests(defaultTests):
    def __init__(
            self, host=jax.devices("cpu")[0], devices=jax.devices(),
            n_per_device=1000, reduced_n_per_device=100, simulate=False,
            **kwargs):
        super().__init__(**kwargs)
        self.host = host
        self.devices = devices
        self.n_devices = len(devices)
        self.n_per_device = n_per_device
        self.reduced_n_per_device = reduced_n_per_device
        self.kwargs["host"] = None
        self.kwargs["devices"] = None
        self.kwargs["n_per_device"] = n_per_device
        self.reduced_kwargs["host"] = None
        self.reduced_kwargs["devices"] = None
        self.reduced_kwargs["n_per_device"] = reduced_n_per_device
        self.simulate = simulate

    def preload(self, dictionary, state=False, validate=False):
        dictionary["host"] = jax.devices("cpu")[0]
        dictionary["devices"] = jax.devices()
        if state:
            dictionary["key_or_state"] = self.state
        if (not self.simulate) and (not validate):
            dictionary.pop("validation_fiducial")
            dictionary.pop("validation_derivative")
        return dictionary

    def specific_exceptions(self, variable, input_variable, kwargs):
        if variable == "validation_fiducial":
            if "validation_fiducial" not in kwargs.keys():
                return True
        if variable == "validation_derivative":
            if "validation_derivative" not in kwargs.keys():
                return True
        if variable == "devices":
            if input_variable is list():
                if len(input_variable) < 1:
                    kwargs[variable] = input_variable
                    with pytest.raises(ValueError) as info:
                        self.imnn(**kwargs)
                    assert info.match("`devices` has no elements in")
                    return True
                if not all(
                        [isinstance(device, jaxlib.xla_extension.Device)
                         for device in input_variable]):
                    kwargs[variable] = input_variable
                    with pytest.raises(TypeError) as info:
                        self.imnn(**kwargs)
                    assert info.match(
                        "`all elements of `devices` must be xla devices")
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
        if variable == "prefetch":
            if input_variable is None:
                return True
        return False

    def initialise_parameters(
            self, variable, kwargs, input_variable, state=False,
            validate=True):
        _kwargs = copy.deepcopy(kwargs)
        _kwargs = self.preload(_kwargs, state=state)

        if variable in _kwargs.keys():
            if variable == "devices":
                valid_type = type(list())
            else:
                valid_type = type(_kwargs[variable])

        if self.exceptions(variable, input_variable, _kwargs):
            return

        if isinstance(input_variable, valid_type):
            return

        _kwargs[variable] = input_variable

        if input_variable is None:
            with pytest.raises(ValueError) as info:
                self.imnn(**_kwargs)
            assert info.match(f"`{variable}` is None")
            return

        with pytest.raises(TypeError) as info:
            self.imnn(**_kwargs)
        assert info.match(f"`{variable}` must be type {valid_type} but " +
                          f"is {type(input_variable)}")
        return

    def splitting(self, variable, kwargs, state=False, validate=True):
        _kwargs = copy.deepcopy(kwargs)
        _kwargs = self.preload(_kwargs, state=state, validate=validate)
        _kwargs["devices"] = [jax.devices()[0]]
        if variable == "same":
            _kwargs["n_s"] = 1000
            _kwargs["n_d"] = 1000
            _kwargs["n_per_device"] = 11
            amount = kwargs["n_s"]
            variable = "n_s and n_d"
        else:
            _kwargs["n_per_device"] = (_kwargs[variable] // 2) - 1
            if variable == "n_s":
                _kwargs["n_d"] = 2 * _kwargs["n_per_device"]
                amount = _kwargs[variable]
            else:
                _kwargs["n_s"] = 2 * _kwargs["n_per_device"]
                amount = _kwargs[variable]
        if not self.simulate:
            _kwargs = self.splitting_resize(_kwargs)

        with pytest.raises(ValueError) as info:
            self.imnn(**_kwargs)
        assert info.match(
            f"`{variable}` of {amount} will not split evenly " +
            f"between {len(_kwargs['devices'])} devices when " +
            f"calculating {_kwargs['n_per_device']} per device.")
        return

    def splitting_resize(self, _kwargs):
        _kwargs["fiducial"] = _kwargs["fiducial"][:_kwargs["n_s"]]
        _kwargs["derivative"] = _kwargs["derivative"][:_kwargs["n_d"]]
        if "validation_fiducial" in _kwargs.keys():
            _kwargs["validation_fiducial"] = \
                _kwargs["validation_fiducial"][:_kwargs["n_s"]]
            _kwargs["validation_derivative"] = \
                _kwargs["validation_derivative"][:_kwargs["n_d"]]
        return _kwargs
