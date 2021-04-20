from test.defaults import defaultTests
import tensorflow as tf
import numpy as np
import datetime
import pathlib
import copy
import pytest
import re

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dense(128),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dense(128),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dense(3)
])
optimiser = tf.keras.optimizers.Adam(learning_rate=1e-3)
false_dataset = tf.data.Dataset.from_tensor_slices(
    tf.constant((1001, 11)))


class tensorflowTests(defaultTests):
    def __init__(
            self, model=model, optimiser=optimiser, model_key=0,
            fit_key=None, stats_key=None, state=None, n_per_batch=1000,
            reduced_n_per_batch=100, cache=False, prefetch=True,
            buffer_size=tf.data.AUTOTUNE, false_dataset=false_dataset,
            **kwargs):
        super().__init__(
            model=model,
            optimiser=optimiser,
            model_key=model_key,
            fit_key=fit_key,
            stats_key=stats_key,
            state=state,
            **kwargs)
        self.print_rate = None
        self.fit_kwargs.pop("print_rate")
        self.n_per_batch = n_per_batch
        self.reduced_n_per_batch = reduced_n_per_batch
        self.cache = cache
        self.prefetch = prefetch
        self.buffer_size = buffer_size
        self.single_target_data = tf.constant(self.single_target_data)
        self.batch_target_date = tf.constant(self.batch_target_data)
        self.kwargs["n_per_batch"] = n_per_batch
        self.kwargs["cache"] = cache
        self.kwargs["prefetch"] = prefetch
        self.kwargs["buffer_size"] = buffer_size
        self.kwargs["θ_fid"] = tf.constant(self.kwargs["θ_fid"])
        self.reduced_kwargs["n_per_batch"] = reduced_n_per_batch
        self.reduced_kwargs["cache"] = cache
        self.reduced_kwargs["prefetch"] = prefetch
        self.reduced_kwargs["buffer_size"] = buffer_size
        self.reduced_kwargs["θ_fid"] = tf.constant(
            self.reduced_kwargs["θ_fid"])
        self.false_dataset = false_dataset
        self.datasets = ["fiducial", "derivative", "validation_fiducial",
                         "validation_derivative"]

    def preload(self, dictionary, state=False, validate=False):
        if "fiducial" in dictionary.keys():
            dictionary["fiducial"] = self.fiducial_dataset
            dictionary["validation_fiducial"] = \
                self.validation_fiducial_dataset
            if dictionary["n_d"] == self.reduced_n_d:
                dictionary["derivative"] = self.reduced_derivative_dataset
                dictionary["validation_derivative"] = \
                    self.reduced_validation_derivative_dataset
            else:
                dictionary["derivative"] = self.derivative_dataset
                dictionary["validation_derivative"] = \
                    self.validation_derivative_dataset
            if (not self.simulate) and (not validate):
                dictionary.pop("validation_fiducial")
                dictionary.pop("validation_derivative")
        return dictionary

    def dataset_type_exception(self, variable, input_variable, kwargs):
        if input_variable is None:
            kwargs[variable] = input_variable
            with pytest.raises(ValueError) as info:
                self.imnn(**kwargs)
            assert info.match(f"`{variable}` is None")
            return True
        elif not isinstance(input_variable, tf.data.Dataset):
            kwargs[variable] = input_variable
            with pytest.raises(TypeError) as info:
                self.imnn(**kwargs)
            assert info.match(f"`{variable}` not from `tf.data.Dataset`")
            return True
        return False

    def dataset_shape_exception(self, variable, input_variable, kwargs):
        shape = kwargs[variable].element_spec.shape
        if shape != input_variable.element_spec.shape:
            kwargs[variable] = input_variable
            with pytest.raises(ValueError) as info:
                self.imnn(**kwargs)
            assert info.match(
                re.escape(
                    f"`{variable}` should have shape {shape} but has " +
                    f"{input_variable.element_spec.shape}"))
            return True
        return False

    def specific_exceptions(self, variable, input_variable, kwargs):
        if variable == "validation_fiducial":
            if "validation_fiducial" not in kwargs.keys():
                return True
        if variable == "validation_derivative":
            if "validation_derivative" not in kwargs.keys():
                return True
        if variable in self.datasets:
            if self.dataset_type_exception(variable, input_variable, kwargs):
                return True
            if self.dataset_shape_exception(variable, input_variable, kwargs):
                return True
        if variable == "n_per_batch":
            if input_variable is None:
                return True
        return False

    def array_type_exception(self, variable, input_variable, kwargs):
        if not isinstance(input_variable, tf.Tensor):
            kwargs[variable] = input_variable
            with pytest.raises(TypeError) as info:
                self.imnn(**kwargs)
            assert info.match(f"`{variable}` must be a `tf.Tensor`")
            return True
        return False

    def optimiser_type_exception(self, input_variable, kwargs):
        if isinstance(input_variable, tf.keras.optimizers.Optimizer):
            return False
        kwargs["optimiser"] = input_variable
        with pytest.raises(TypeError) as info:
            self.imnn(**kwargs)
        assert info.match(
            "`optimiser` not from `tf.keras.optimizers`")
        return True

    def optimiser_length_exception(self, input_variable, kwargs):
        return False

    def optimiser_call_exception(self, input_variable, kwargs):
        return False

    def model_type_exception(self, input_variable, kwargs):
        if isinstance(input_variable, tf.keras.models.Model):
            return False
        kwargs["model"] = input_variable
        with pytest.raises(TypeError) as info:
            self.imnn(**kwargs)
        assert info.match(
            "`model` not from `tf.keras.models`")
        return True

    def model_length_exception(self, input_variable, kwargs):
        return False

    def model_call_exception(self, input_variable, kwargs):
        return False

    def state_exception(self, input_variable, kwargs):
        return True

    def splitting(self, variable, kwargs, state=False, validate=True):
        _kwargs = copy.deepcopy(kwargs)
        _kwargs = self.preload(_kwargs, state=state, validate=validate)
        if variable == "same":
            _kwargs["n_s"] = 1000
            _kwargs["n_d"] = 1000
            _kwargs["n_per_batch"] = 11
            amount = kwargs["n_s"]
            variable = "n_s and n_d"
        else:
            _kwargs["n_per_batch"] = (_kwargs[variable] // 2) - 1
            if variable == "n_s":
                _kwargs["n_d"] = 2 * _kwargs["n_per_batch"]
                amount = _kwargs[variable]
            else:
                _kwargs["n_s"] = 2 * _kwargs["n_per_batch"]
                amount = _kwargs[variable]
        if "fiducial" in _kwargs.keys():
            _kwargs["fiducial"] = tf.data.Dataset.from_tensor_slices(
                self.fiducial[:_kwargs["n_s"]])
            _kwargs["derivative"] = tf.data.Dataset.from_tensor_slices(
                self.derivative[:_kwargs["n_d"]])
        if "validation_fiducial" in _kwargs.keys():
            _kwargs["validation_fiducial"] = \
                tf.data.Dataset.from_tensor_slices(
                    self.validation_fiducial[:_kwargs["n_s"]])
            _kwargs["validation_derivative"] = \
                tf.data.Dataset.from_tensor_slices(
                    self.validation_derivative[:_kwargs["n_s"]])

        with pytest.raises(ValueError) as info:
            self.imnn(**_kwargs)
        assert info.match(
            f"`{variable}` of {amount} will not split evenly when " +
            f"calculating {_kwargs['n_per_batch']} per batch")
        return

    def fit(self, kwargs, fit_kwargs, state=False, validate=True, set=True):
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
        imnn.fit(λ, ϵ, **_fit_kwargs)

        name = self.set_name(state, validate, False, _kwargs["n_d"])
        if self.save:
            files = {
                f"{name}F": imnn.F.numpy(),
                f"{name}C": imnn.C.numpy(),
                f"{name}invC": imnn.invC.numpy(),
                f"{name}dμ_dθ": imnn.dμ_dθ.numpy(),
                f"{name}μ": imnn.μ.numpy(),
                f"{name}invF": imnn.invF.numpy(),
            }
            try:
                targets = np.load(f"test/{self.filename}.npz")
                files = {k: files[k] for k in files.keys() - targets.keys()}
                np.savez(f"test/{self.filename}.npz", **{**files, **targets})
            except Exception:
                np.savez(f"test/{self.filename}.npz", **files)
        targets = np.load(f"test/{self.filename}.npz")
        assert np.all(np.equal(imnn.F.numpy(), targets[f"{name}F"]))
        assert np.all(np.equal(imnn.C.numpy(), targets[f"{name}C"]))
        assert np.all(np.equal(imnn.invC.numpy(), targets[f"{name}invC"]))
        assert np.all(np.equal(imnn.dμ_dθ.numpy(), targets[f"{name}dμ_dθ"]))
        assert np.all(np.equal(imnn.μ.numpy(), targets[f"{name}μ"]))
        assert np.all(np.equal(imnn.invF.numpy(), targets[f"{name}invF"]))

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
            files = {f"{name}estimate": estimate.numpy()}
            try:
                targets = np.load(f"test/{self.filename}.npz")
                files = {k: files[k] for k in files.keys() - targets.keys()}
                np.savez(f"test/{self.filename}.npz", **{**files, **targets})
            except Exception:
                np.savez(f"test/{self.filename}.npz", **files)
        targets = np.load(f"test/{self.filename}.npz")
        assert np.all(np.equal(estimate.numpy(), targets[f"{name}estimate"]))

    def training_plot(
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
            self.imnn(**_kwargs).training_plot(
                expected_detF=50,
                filename=f"test/figures/{self.filename}/{name}_{time}.pdf")
            assert pathlib.Path(
                f"test/figures/{self.filename}/{name}_{time}.pdf").is_file()

    def combined_running_test(
            self, data, kwargs, fit_kwargs, state=False, validate=True,
            fit=False, implemented=True):
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
                imnn.fit(λ, ϵ, **_fit_kwargs)
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
                estimates.append(imnn.get_estimate(element).numpy())

            if self.save:
                files = {
                    f"{name}estimates": estimates,
                    f"{name}F": imnn.F.numpy(),
                    f"{name}C": imnn.C.numpy(),
                    f"{name}invC": imnn.invC.numpy(),
                    f"{name}dμ_dθ": imnn.dμ_dθ.numpy(),
                    f"{name}μ": imnn.μ.numpy(),
                    f"{name}invF": imnn.invF.numpy()}
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
            assert np.all(np.equal(imnn.F.numpy(), targets[f"{name}F"]))
            assert np.all(np.equal(imnn.C.numpy(), targets[f"{name}C"]))
            assert np.all(np.equal(imnn.invC.numpy(), targets[f"{name}invC"]))
            assert np.all(np.equal(
                imnn.dμ_dθ.numpy(), targets[f"{name}dμ_dθ"]))
            assert np.all(np.equal(imnn.μ.numpy(), targets[f"{name}μ"]))
            assert np.all(np.equal(imnn.invF.numpy(), targets[f"{name}invF"]))

        imnn.training_plot(
            expected_detF=50,
            filename=f"test/figures/{self.filename}/{name}_{time}.pdf")
        assert pathlib.Path(
            f"test/figures/{self.filename}/{name}_{time}.pdf").is_file()
