import tensorflow as tf
from test.tensorflow_defaults import tensorflowTests
from test.numerical_gradient_defaults import numericalGradientTests


class tensorflowNumericalGradientTests(
        tensorflowTests, numericalGradientTests):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.δθ = tf.constant(self.δθ)
        self.fiducial_dataset = tf.data.Dataset.from_tensor_slices(
            self.fiducial)
        self.derivative_dataset = tf.data.Dataset.from_tensor_slices(
            self.derivative)
        self.reduced_derivative_dataset = tf.data.Dataset.from_tensor_slices(
            self.reduced_derivative)
        self.validation_fiducial_dataset = tf.data.Dataset.from_tensor_slices(
            self.validation_fiducial)
        self.validation_derivative_dataset = \
            tf.data.Dataset.from_tensor_slices(self.validation_derivative)
        self.reduced_validation_derivative_dataset = \
            tf.data.Dataset.from_tensor_slices(
                self.reduced_validation_derivative)
        self.arrays = ["θ_fid", "δθ"]
        self.kwargs["δθ"] = self.δθ
        self.kwargs["fiducial"] = None
        self.kwargs["derivative"] = None
        self.kwargs["validation_fiducial"] = None
        self.kwargs["validation_derivative"] = None
        self.reduced_kwargs["δθ"] = self.δθ
        self.reduced_kwargs["fiducial"] = None
        self.reduced_kwargs["derivative"] = None
        self.reduced_kwargs["validation_fiducial"] = None
        self.reduced_kwargs["validation_derivative"] = None
