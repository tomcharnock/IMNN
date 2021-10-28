import pytest
import copy
import pathlib
import re
import inspect
import datetime
import os
import jax
import jax.numpy as np
import numpy as onp
import jax.experimental.optimizers as optimizers
import jax.experimental.stax as stax
import matplotlib.pyplot as plt
from imnn.utils import value_and_jacfwd

rng = jax.random.PRNGKey(0)
rng, model_key, data_key, fit_key, stats_key = jax.random.split(rng, num=5)
model = stax.serial(
    stax.Dense(128),
    stax.LeakyRelu,
    stax.Dense(128),
    stax.LeakyRelu,
    stax.Dense(128),
    stax.LeakyRelu,
    stax.Dense(3),)
bad_model = stax.serial(
    stax.Dense(5),)
optimiser = optimizers.adam(step_size=1e-3)
_, initial_w = model[0](model_key, (10,))
state = optimiser[0](initial_w)


def simulator(rng, θ):
    return θ[0] + np.sqrt(θ[1]) * jax.random.normal(rng, shape=(10,))


def simulator_gradient(rng, θ):
    return value_and_jacfwd(simulator, argnums=1)(rng, θ)


class defaultTests:
    def __init__(
            self, rng=rng, model_key=model_key, data_key=data_key,
            fit_key=fit_key, stats_key=stats_key, n_s=1000, n_d=1000,
            reduced_n_d=100, n_params=2, n_summaries=3, input_shape=(10,),
            θ_fid=np.array([0., 1.]), model=model, optimiser=optimiser,
            state=state, λ=10., ϵ=0.01, patience=5, min_iterations=1,
            max_iterations=10, print_rate=None, best=True, imnn=None,
            filename=None, force_save=False):
        self.rng = rng
        self.model_key = model_key
        self.data_key = data_key
        self.fit_key = fit_key
        self.stats_key = stats_key
        self.n_s = n_s
        self.n_d = n_d
        self.reduced_n_d = reduced_n_d
        self.n_params = n_params
        self.n_summaries = n_summaries
        self.input_shape = input_shape
        self.θ_fid = θ_fid
        self.model = model
        self.bad_model = bad_model
        self.optimiser = optimiser
        self.state = state
        self.kwargs = {
            "n_s": n_s,
            "n_d": n_d,
            "n_params": n_params,
            "n_summaries": n_summaries,
            "input_shape": input_shape,
            "θ_fid": θ_fid,
            "model": model,
            "optimiser": optimiser,
            "key_or_state": model_key}
        self.reduced_kwargs = copy.deepcopy(self.kwargs)
        self.reduced_kwargs["n_d"] = reduced_n_d
        self.λ = λ
        self.ϵ = ϵ
        self.patience = patience
        self.min_iterations = min_iterations
        self.max_iterations = max_iterations
        self.print_rate = print_rate
        self.best = best
        self.fit_kwargs = {
            "λ": λ,
            "ϵ": ϵ,
            "rng": fit_key,
            "patience": patience,
            "min_iterations": min_iterations,
            "max_iterations": max_iterations,
            "print_rate": print_rate,
            "best": best}
        self.arrays = ["θ_fid"]
        self.single_target_data = jax.random.normal(data_key, input_shape)
        self.batch_target_data = np.expand_dims(self.single_target_data, 0)
        self.imnn = imnn
        self.simulate = False
        self.filename = filename
        if filename is not None:
            if not os.path.isfile(f"test/{filename}.npz"):
                self.save = True
            else:
                self.save = False
        if force_save:
            self.save = True

    def preload(self, dictionary, state=False, validate=False):
        if state:
            dictionary["key_or_state"] = self.state
        return dictionary

    def set_name(self, state, validate, fit, size, name=""):
        if size == self.reduced_n_d:
            name = f"reduced_{name}"
        if fit:
            name = f"fit_{name}"
        if state:
            name = f"from_state_{name}"
        if validate:
            name = f"validate_{name}"
        return name

    def array_type_exception(self, variable, input_variable, kwargs):
        if not isinstance(
            input_variable,
            (np.ndarray, jax.interpreters.xla._DeviceArray, onp.ndarray)):
            kwargs[variable] = input_variable
            with pytest.raises(TypeError) as info:
                self.imnn(**kwargs)
            assert info.match(f"`{variable}` must be a jax array")
            return True
        return False

    def array_shape_exception(self, variable, input_variable, kwargs):
        valid_shape = kwargs[variable].shape
        if input_variable.shape != valid_shape:
            kwargs[variable] = input_variable
            with pytest.raises(ValueError) as info:
                self.imnn(**kwargs)
            assert info.match(
                re.escape(
                    f"`{variable}` should have shape {valid_shape} but has " +
                    f"{input_variable.shape}"))
            return True
        return False

    def array_exception(self, variable, input_variable, kwargs):
        if input_variable is None:
            return False
        if self.array_type_exception(variable, input_variable, kwargs):
            return True
        if self.array_shape_exception(variable, input_variable, kwargs):
            return True
        return False

    def allow_None_state(self, input_variable, kwargs):
        if input_variable is None:
            kwargs["key_or_state"] = input_variable
            with pytest.raises(ValueError) as info:
                self.imnn(**kwargs)
            assert info.match("`key_or_state` is None")
            return True
        return False

    def state_type_exception(self, input_variable):
        if isinstance(
            input_variable,
            (jax.interpreters.xla.device_array, np.ndarray, onp.ndarray)):
            if input_variable.shape == (2,):
                return True
        if isinstance(input_variable,
                      jax.experimental.optimizers.OptimizerState):
            return True
        return False

    def state_callable_exception(self, input_variable, kwargs):
        try:
            self.optimiser[2](input_variable)
        except Exception:
            kwargs["key_or_state"] = input_variable
            with pytest.raises(TypeError) as info:
                self.imnn(**kwargs)
            assert info.match(
                "`state` is not valid for extracting parameters from")
        return True

    def state_exception(self, input_variable, kwargs):
        if self.allow_None_state(input_variable, kwargs):
            return True
        if self.state_type_exception(input_variable):
            return True
        if self.state_callable_exception(input_variable, kwargs):
            return True

    def optimiser_type_exception(self, input_variable, kwargs):
        return False

    def optimiser_length_exception(self, input_variable, kwargs):
        def test(input_variable, kwargs):
            kwargs["optimiser"] = input_variable
            with pytest.raises(TypeError) as info:
                self.imnn(**kwargs)
            assert info.match(
                "`optimiser` must be tuple of three functions. The first " +
                "for initialising the state, the second to update the state " +
                "and the third to get parameters from the state.")
            return True
        try:
            length = len(input_variable)
        except Exception:
            if test(input_variable, kwargs):
                return True
        if length != 3:
            if test(input_variable, kwargs):
                return True
        return False

    def optimiser_call_exception(self, input_variable, kwargs):
        def test(input_variable, kwargs, element):
            errors = [
                "first element of `optimiser` must take one argument",
                "second element of `optimiser` must take three arguments",
                "third element of `optimiser` must take one argument"]
            lengths = [1, 3, 1]
            kwargs["optimiser"] = input_variable
            try:
                inspect.signature(input_variable[element]).parameters
            except Exception:
                with pytest.raises(TypeError) as info:
                    self.imnn(**kwargs)
                assert info.match(errors[element])
                return True
            if len(inspect.signature(
                    input_variable[element]).parameters) != lengths[element]:
                with pytest.raises(ValueError) as info:
                    self.imnn(**kwargs)
                assert info.match(errors[element])
                return True
            return False
        for element in range(3):
            if test(input_variable, kwargs, element):
                return True
        return False

    def optimiser_exception(self, input_variable, kwargs):
        if input_variable is None:
            return False
        if self.optimiser_type_exception(input_variable, kwargs):
            return True
        if self.optimiser_length_exception(input_variable, kwargs):
            return True
        if self.optimiser_call_exception(input_variable, kwargs):
            return True
        return False

    def model_type_exception(self, input_variable, kwargs):
        if isinstance(input_variable, tuple):
            return False
        kwargs["model"] = input_variable
        with pytest.raises(TypeError) as info:
            self.imnn(**kwargs)
        assert info.match(
            f"`model` must be type {tuple} but is {type(input_variable)}")
        return True

    def model_length_exception(self, input_variable, kwargs):
        if len(input_variable) != 2:
            kwargs["model"] = input_variable
            with pytest.raises(ValueError) as info:
                self.imnn(**kwargs)
            assert info.match(
                "`model` must be a tuple of two functions. The first for " +
                "initialising the model and the second to call the model")
            return True
        return False

    def model_call_exception(self, input_variable, kwargs):
        def test(input_variable, kwargs, element):
            errors = [
                "first element of `model` must take two arguments",
                "second element of `model` must take three arguments"]
            lengths = [2, 3]
            kwargs["model"] = input_variable
            try:
                inspect.signature(input_variable[element]).parameters
            except Exception:
                with pytest.raises(TypeError) as info:
                    self.imnn(**kwargs)
                assert info.match(errors[element])
                return True
            if len(inspect.signature(
                    input_variable[element]).parameters) != lengths[element]:
                with pytest.raises(ValueError) as info:
                    self.imnn(**kwargs)
                assert info.match(errors[element])
                return True
            return False
        for element in range(2):
            if test(input_variable, kwargs, element):
                return True
        return False

    def model_initialisation_exception(self, input_variable, kwargs):
        if ((kwargs["key_or_state"] == self.model_key)
                and (kwargs["n_summaries"] == self.n_summaries)):
            output_shape, _ = input_variable[0](
                kwargs["key_or_state"], (kwargs["n_summaries"],))
            if output_shape != (self.n_summaries,):
                with pytest.raises(ValueError) as info:
                    self.imnn(**kwargs)
                assert info.match(
                    "`model` outputs should have shape " +
                    f"{(self.n_summaries,)} but is {output_shape}")
                return True
        return False

    def model_exception(self, input_variable, kwargs):
        if input_variable is None:
            return False
        if self.model_type_exception(input_variable, kwargs):
            return True
        if self.model_length_exception(input_variable, kwargs):
            return True
        if self.model_call_exception(input_variable, kwargs):
            return True
        return False

    def specific_exceptions(self, variable, input_variable, kwargs):
        return False

    def exceptions(self, variable, input_variable, kwargs):
        if self.specific_exceptions(variable, input_variable, kwargs):
            return True
        if variable in self.arrays:
            if self.array_exception(variable, input_variable, kwargs):
                return True
        if variable == "key_or_state":
            if self.state_exception(input_variable, kwargs):
                return True
        if variable == "optimiser":
            if self.optimiser_exception(input_variable, kwargs):
                return True
        if variable == "model":
            if self.model_exception(input_variable, kwargs):
                return True
        return False

    def initialise_parameters(
            self, variable, kwargs, input_variable, state=False,
            validate=True):
        _kwargs = copy.deepcopy(kwargs)
        _kwargs = self.preload(_kwargs, state=state)

        if variable in _kwargs.keys():
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

    def rng_fit_type_exception(self, input_variable, kwargs, fit_kwargs):
        if not isinstance(
            input_variable,
            (jax.interpreters.xla._DeviceArray, np.ndarray, onp.ndarray)):
            fit_kwargs["rng"] = input_variable
            λ = fit_kwargs.pop("λ")
            ϵ = fit_kwargs.pop("ϵ")
            with pytest.raises(TypeError) as info:
                self.imnn(**kwargs).fit(λ, ϵ, **fit_kwargs)
            assert info.match("`rng` must be a jax array")
            return True
        return False

    def rng_fit_shape_exception(self, input_variable, kwargs, fit_kwargs):
        if fit_kwargs["rng"] is None:
            return True
        valid_shape = fit_kwargs["rng"].shape
        if input_variable.shape != valid_shape:
            fit_kwargs["rng"] = input_variable
            λ = fit_kwargs.pop("λ")
            ϵ = fit_kwargs.pop("ϵ")
            with pytest.raises(ValueError) as info:
                self.imnn(**kwargs).fit(λ, ϵ, **fit_kwargs)
            assert info.match(
                re.escape(
                    f"`rng` should have shape {valid_shape} but has " +
                    f"{input_variable.shape}"))
            return True
        return False

    def rng_fit_exception(self, input_variable, kwargs, fit_kwargs):
        if input_variable is None:
            return True
        if self.rng_fit_type_exception(input_variable, kwargs, fit_kwargs):
            return True
        if self.rng_fit_shape_exception(input_variable, kwargs, fit_kwargs):
            return True
        return False

    def print_rate_fit_type_exception(
            self, input_variable, kwargs, fit_kwargs):
        if not isinstance(input_variable, int):
            fit_kwargs["print_rate"] = input_variable
            λ = fit_kwargs.pop("λ")
            ϵ = fit_kwargs.pop("ϵ")
            with pytest.raises(TypeError) as info:
                self.imnn(**kwargs).fit(λ, ϵ, **fit_kwargs)
            assert info.match(
                f"`print_rate` must be type {int} but is " +
                f"{type(input_variable)}")
            return True
        return True

    def print_rate_fit_exception(self, input_variable, kwargs, fit_kwargs):
        if input_variable is None:
            return True
        if isinstance(input_variable, int):
            return True
        if self.print_rate_fit_type_exception(
                input_variable, kwargs, fit_kwargs):
            return True
        return False

    def fit_exceptions(self, variable, input_variable, kwargs, fit_kwargs):
        if variable == "rng":
            if self.rng_fit_exception(input_variable, kwargs, fit_kwargs):
                return True
        if variable == "print_rate":
            if self.print_rate_fit_exception(
                    input_variable, kwargs, fit_kwargs):
                return True
        return False

    def fit_parameters(
            self, variable, kwargs, fit_kwargs, input_variable, state=False,
            validate=True):
        _kwargs = copy.deepcopy(kwargs)
        _kwargs = self.preload(_kwargs, state=state, validate=validate)
        _fit_kwargs = copy.deepcopy(fit_kwargs)

        valid_type = type(_fit_kwargs[variable])

        if self.fit_exceptions(variable, input_variable, _kwargs, _fit_kwargs):
            return

        if isinstance(input_variable, valid_type):
            return

        _fit_kwargs[variable] = input_variable
        λ = _fit_kwargs.pop("λ")
        ϵ = _fit_kwargs.pop("ϵ")

        if input_variable is None:
            with pytest.raises(ValueError) as info:
                self.imnn(**_kwargs).fit(λ, ϵ, **_fit_kwargs)
            assert info.match(f"`{variable}` is None")
            return

        with pytest.raises(TypeError) as info:
            self.imnn(**_kwargs).fit(λ, ϵ, **_fit_kwargs)
        assert info.match(
            f"`{variable}` must be type {valid_type} but is " +
            f"{type(input_variable)}")
        return

    def fit(self, kwargs, fit_kwargs, state=False, validate=True, set=True,
            none_first=True):
        _kwargs = copy.deepcopy(kwargs)
        _kwargs = self.preload(_kwargs, state=state, validate=validate)
        _fit_kwargs = copy.deepcopy(fit_kwargs)
        λ = _fit_kwargs.pop("λ")
        ϵ = _fit_kwargs.pop("ϵ")

        imnn = self.imnn(**_kwargs)
        if not set:
            with pytest.raises(ValueError) as info:
                self.imnn(**_kwargs).fit(λ, ϵ, **_fit_kwargs)
            assert info.match("`get_summaries` not implemented")
            return

        if none_first:
            _fit_kwargs["print_rate"] = None
            string = ("Cannot run IMNN with progress bar after running " +
                      "without progress bar. Either set `print_rate` to " +
                      "None or reinitialise the IMNN.")
        else:
            _fit_kwargs["print_rate"] = 100
            string = ("Cannot run IMNN without progress bar after running " +
                      "with progress bar. Either set `print_rate` to an int " +
                      "or reinitialise the IMNN.")

        imnn.fit(λ, ϵ, **_fit_kwargs)

        name = self.set_name(state, validate, False, _kwargs["n_d"])
        if self.save:
            files = {
                f"{name}F": imnn.F,
                f"{name}C": imnn.C,
                f"{name}invC": imnn.invC,
                f"{name}dμ_dθ": imnn.dμ_dθ,
                f"{name}μ": imnn.μ,
                f"{name}invF": imnn.invF,
            }
            try:
                targets = np.load(f"test/{self.filename}.npz")
                files = {k: files[k] for k in files.keys() - targets.keys()}
                np.savez(f"test/{self.filename}.npz", **{**files, **targets})
            except Exception:
                np.savez(f"test/{self.filename}.npz", **files)
        targets = np.load(f"test/{self.filename}.npz")
        assert np.all(np.equal(imnn.F, targets[f"{name}F"]))
        assert np.all(np.equal(imnn.C, targets[f"{name}C"]))
        assert np.all(np.equal(imnn.invC, targets[f"{name}invC"]))
        assert np.all(np.equal(imnn.dμ_dθ, targets[f"{name}dμ_dθ"]))
        assert np.all(np.equal(imnn.μ, targets[f"{name}μ"]))
        assert np.all(np.equal(imnn.invF, targets[f"{name}invF"]))

        if none_first:
            _fit_kwargs["print_rate"] = 100
        else:
            _fit_kwargs["print_rate"] = None

        with pytest.raises(ValueError) as info:
            imnn(**_kwargs).fit(λ, ϵ, **_fit_kwargs)
        assert info.match(string)
        return

    def get_estimate(
            self, data, kwargs, fit_kwargs=None, state=False, validate=True,
            fit=False, set=True):
        _kwargs = copy.deepcopy(kwargs)
        _kwargs = self.preload(_kwargs, state=state, validate=validate)

        imnn = self.imnn(**_kwargs)

        if not set:
            with pytest.raises(ValueError) as info:
                imnn.get_estimate(data)
            assert info.match(
                re.escape(
                    "Fisher information has not yet been calculated. Please " +
                    "run `imnn.set_F_statistics({w}, {key}, {validate}) " +
                    "with `w = imnn.final_w`, `w = imnn.best_w`, " +
                    "`w = imnn.inital_w` or otherwise, `validate = True` " +
                    "should be set if not simulating on the fly."))
            return

        _fit_kwargs = copy.deepcopy(fit_kwargs)
        λ = _fit_kwargs.pop("λ")
        ϵ = _fit_kwargs.pop("ϵ")
        if fit:
            imnn.fit(λ, ϵ, **_fit_kwargs)
        else:
            imnn.set_F_statistics(key=self.stats_key)
        estimate = imnn.get_estimate(data)

        name = self.set_name(state, validate, fit, _kwargs["n_d"])
        if self.save:
            files = {f"{name}estimate": estimate}
            try:
                targets = np.load(f"test/{self.filename}.npz")
                files = {k: files[k] for k in files.keys() - targets.keys()}
                np.savez(f"test/{self.filename}.npz", **{**files, **targets})
            except Exception:
                np.savez(f"test/{self.filename}.npz", **files)
        targets = np.load(f"test/{self.filename}.npz")
        assert np.all(np.equal(estimate, targets[f"{name}estimate"]))

    def plot(
            self, kwargs, fit_kwargs, state=False, validate=True, fit=False,
            set=True):
        _kwargs = copy.deepcopy(kwargs)
        _kwargs = self.preload(_kwargs, state=state, validate=validate)

        time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        name = self.set_name(state, validate, fit, _kwargs["n_d"], name="plot")

        imnn = self.imnn(**_kwargs)

        if fit:
            _fit_kwargs = copy.deepcopy(fit_kwargs)
            λ = _fit_kwargs.pop("λ")
            ϵ = _fit_kwargs.pop("ϵ")
            imnn.fit(λ, ϵ, **_fit_kwargs)

        if self.filename is not None:
            self.imnn(**_kwargs).plot(
                expected_detF=50,
                filename=f"test/figures/{self.filename}/{name}_{time}.pdf")
            assert pathlib.Path(
                f"test/figures/{self.filename}/{name}_{time}.pdf").is_file()

    def combined_running_test(
            self, data, kwargs, fit_kwargs, state=False, validate=True,
            fit=False, none_first=True, implemented=True, aggregated=False):
        _kwargs = copy.deepcopy(kwargs)
        _kwargs = self.preload(_kwargs, state=state, validate=validate)

        imnn = self.imnn(**_kwargs)

        with pytest.raises(ValueError) as info:
            imnn.get_estimate(data)
        assert info.match(
            re.escape(
                "Fisher information has not yet been calculated. Please " +
                "run `imnn.set_F_statistics({w}, {key}, {validate}) " +
                "with `w = imnn.final_w`, `w = imnn.best_w`, " +
                "`w = imnn.inital_w` or otherwise, `validate = True` " +
                "should be set if not simulating on the fly."))

        time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        name = self.set_name(state, validate, fit, _kwargs["n_d"])

        if fit:
            _fit_kwargs = copy.deepcopy(fit_kwargs)
            λ = _fit_kwargs.pop("λ")
            ϵ = _fit_kwargs.pop("ϵ")

            if not implemented:
                with pytest.raises(ValueError) as info:
                    imnn.fit(λ, ϵ, **_fit_kwargs)
                assert info.match("`get_summaries` not implemented")
            else:
                if none_first:
                    _fit_kwargs["print_rate"] = None
                    string = ("Cannot run IMNN with progress bar after " +
                              "running without progress bar. Either set " +
                              "`print_rate` to None or reinitialise the IMNN.")
                else:
                    _fit_kwargs["print_rate"] = 100
                    string = ("Cannot run IMNN without progress bar after " +
                              "running with progress bar. Either set " +
                              "`print_rate` to an int or reinitialise the " +
                              "IMNN.")
                imnn.fit(λ, ϵ, **_fit_kwargs)
                if none_first:
                    _fit_kwargs["print_rate"] = 100
                else:
                    _fit_kwargs["print_rate"] = None
        else:
            if not implemented:
                with pytest.raises(ValueError) as info:
                    imnn.set_F_statistics(key=self.stats_key)
                assert info.match("`get_summaries` not implemented")
            else:
                imnn.set_F_statistics(key=self.stats_key)

        if implemented:
            estimates = []
            for element in data:
                estimates.append(imnn.get_estimate(element))

            if self.save:
                files = {
                    f"{name}estimates": estimates,
                    f"{name}F": imnn.F,
                    f"{name}C": imnn.C,
                    f"{name}invC": imnn.invC,
                    f"{name}dμ_dθ": imnn.dμ_dθ,
                    f"{name}μ": imnn.μ,
                    f"{name}invF": imnn.invF}
                try:
                    targets = np.load(f"test/{self.filename}.npz")
                    files = {
                        k: files[k] for k in files.keys() - targets.keys()}
                    np.savez(f"test/{self.filename}.npz",
                             **{**files, **targets},
                             allow_pickle=True)
                except Exception:
                    np.savez(f"test/{self.filename}.npz",
                             **files,
                             allow_pickle=True)
            targets = np.load(f"test/{self.filename}.npz", allow_pickle=True)
            for i, estimate in enumerate(estimates):
                assert np.all(
                    np.equal(estimate, targets[f"{name}estimates"][i]))
            assert np.all(np.equal(imnn.F, targets[f"{name}F"]))
            assert np.all(np.equal(imnn.C, targets[f"{name}C"]))
            assert np.all(np.equal(imnn.invC, targets[f"{name}invC"]))
            assert np.all(np.equal(imnn.dμ_dθ, targets[f"{name}dμ_dθ"]))
            assert np.all(np.equal(imnn.μ, targets[f"{name}μ"]))
            assert np.all(np.equal(imnn.invF, targets[f"{name}invF"]))

        imnn.plot(
            expected_detF=50,
            filename=f"test/figures/{self.filename}/{name}_{time}.pdf")
        plt.close("all")
        assert pathlib.Path(
            f"test/figures/{self.filename}/{name}_{time}.pdf").is_file()

        if fit and implemented and (not aggregated):
            with pytest.raises(ValueError) as info:
                imnn.fit(λ, ϵ, **_fit_kwargs)
            assert info.match(string)
