import tensorflow as tf


def _check_tensorflow(input, input_type, name):
    if input is None:
        raise ValueError(f"`{name}` is None")
    elif not isinstance(input, input_type):
        if name == "optimiser":
            string = f"`{name}` not from `tf.keras.optimizers`"
        elif name == "model":
            string = f"`{name}` not from `tf.keras.models`"
        else:
            string = f"`{name}` must be a `tf.Tensor`"
        raise TypeError(string)
    return input


def _check_tensorflow_input(input, shape, name):
    input = _check_tensorflow(input, tf.Tensor, name)
    if input.shape != shape:
        raise ValueError(f"`{name}` should have shape {shape} but has " +
                         f"{input.shape}")
    return input


def _check_dataset(input, shape, name):
    if input is None:
        raise ValueError(f"`{name}` is None")
    elif not isinstance(input, tf.data.Dataset):
        raise TypeError(f"`{name}` not from `tf.data.Dataset`")
    elif input.element_spec.shape != tf.TensorShape(shape):
        raise ValueError(
            f"`{name}` should have shape {tf.TensorShape(shape)} but has " +
            f"{input.element_spec.shape}")
    return input


def _check_batch_splitting(size, name, n_per_batch):
    if (size / n_per_batch
            != float(size // n_per_batch)):
        raise ValueError(f"`{name}` of {size} will not split evenly " +
                         f"when calculating {n_per_batch} per batch")
