import tensorflow as tf
import numpy as np
from inspect import signature
import os
import sys


class TFRecords():
    """ Module for writing simulations to TFRecord to be used by the IMNN

    Attributes
    __________
    record_size : int
        approximate maximum size of an individual record (in Mb)
    padding : int
        zero padding size for records
    input_shape : tuple
        shape of a single simulation
    """
    def __init__(self, record_size=150, padding=5):
        self.record_size = record_size * int(1e6)
        self.padding = padding
        self.input_shape = None

    def write_record(
            self, n_sims, get_simulation, fiducial=True, n_params=None,
            validation=False, directory=None, filename=None, start=0):
        """Write all simulations to set of tfrecords

        Parameters
        __________
        n_sims : int
            number of simulations to be written to record
        get_simulation : func
            function (1 or 3 inputs for fiducial or derivative) which returns a
            single simulation as a numpy array
        fiducial : bool (default True)
            whether the simulations are in the fiducial or derivative format
        n_params : int (opt)
            number of parameters in the simulator model (for the derivative)
        validation : bool (default False)
            tag to automatically prepend `validation` to the filename
        directory : str (opt)
            directory to save records. defaults to current directory
        filename : str (opt)
            filename to save records. defaults to `fiducial` and `derivative`
            depending on the value of `fiducial`
        start : int (opt)
            value to start seed at
        """
        self.check_func(get_simulation, fiducial)
        self.check_params(n_params, fiducial)
        self.file = self.get_file(directory, filename, fiducial, validation)
        serialise = self.get_serialiser(fiducial, get_simulation, n_params)
        record = True
        counter = 0
        simulation = self.get_initial_seed(fiducial, start)
        while record:
            with tf.io.TFRecordWriter(".".join((
                    "_".join((
                        self.file,
                        "{}".format(counter).zfill(self.padding))),
                    "tfrecords"))) as self.writer:
                while self.get_seed(simulation, fiducial) < n_sims:
                    simulation = serialise(simulation, counter)
                    if self.check_size(counter):
                        counter += 1
                        break
            if self.get_seed(simulation, fiducial) == n_sims:
                record = False

    def fiducial_serialiser(self, seed, counter, get_simulation):
        print("seed={}, record={}".format(seed, counter), end="\r")
        data = get_simulation(seed)
        if self.input_shape is None:
            self.input_shape = data.shape
        data = data.astype(np.float32)
        data = data.tostring()
        feature = {"data": self._bytes_feature(data),
                   "seed": self._int64_feature(seed)}
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        self.writer.write(example.SerializeToString())
        seed += 1
        return seed

    def derivative_serialiser(
            self, simulation, counter, get_simulation, n_params):
        seed, derivative, parameter = simulation
        break_out = False
        while derivative < 2:
            while parameter < n_params:
                print("seed={}, derivative={}, parameter={}, record={}".format(
                    seed, derivative, parameter, counter), end="\r")
                data = get_simulation(seed, derivative, parameter)
                data = data.astype(np.float32)
                data = data.tostring()
                feature = {"data": self._bytes_feature(data),
                           "seed": self._int64_feature(seed),
                           "derivative": self._int64_feature(derivative),
                           "parameter": self._int64_feature(parameter)}
                example = tf.train.Example(
                    features=tf.train.Features(feature=feature))
                self.writer.write(example.SerializeToString())
                parameter += 1
                if self.check_size(counter):
                    break_out = True
                    break
                else:
                    break_out = False
            if break_out:
                break
            else:
                derivative += 1
                parameter = 0
        if not break_out:
            seed += 1
            derivative = 0
        return (seed, derivative, parameter)

    def get_serialiser(self, fiducial, get_simulation, n_params):
        if fiducial:
            return lambda inds, counter: self.fiducial_serialiser(
                inds, counter, get_simulation)
        else:
            return lambda inds, counter: self.derivative_serialiser(
                inds, counter, get_simulation, n_params)

    def parser(self, example):
        parsed_example = tf.io.parse_single_example(
            example, {"data": tf.io.FixedLenFeature([], tf.string)})
        return tf.reshape(
            tf.io.decode_raw(parsed_example["data"], tf.float32),
            self.input_shape)

    def derivative_parser(self, example, n_params=None):
        parsed_example = tf.io.parse_single_example(
            example, {"data": tf.io.FixedLenFeature([], tf.string)})
        return tf.reshape(
            tf.io.decode_raw(parsed_example["data"], tf.float32),
            self.input_shape + (n_params,))

    def numerical_derivative_parser(self, example, n_params=None):
        parsed_example = tf.io.parse_single_example(
            example, {"data": tf.io.FixedLenFeature([], tf.string)})
        return tf.reshape(
            tf.io.decode_raw(parsed_example["data"], tf.float32),
            (2, n_params) + self.input_shape)

    def get_file(self, directory, filename, fiducial, validation):
        if filename is None:
            if fiducial:
                filename = "fiducial"
            else:
                filename = "derivative"
            if validation:
                filename = "validation_{}".format(filename)
        if directory is not None:
            file = "/".join((directory, filename))
        else:
            file = filename
        return file

    def check_size(self, counter):
        return os.path.getsize(
            ".".join((
                "_".join((
                    self.file,
                    "{}".format(counter).zfill(self.padding))),
                "tfrecords"))) > self.record_size

    def get_initial_seed(self, fiducial, start):
        if fiducial:
            return start
        else:
            return (start, 0, 0)

    def get_seed(self, simulation, fiducial):
        if fiducial:
            return simulation
        else:
            return simulation[0]

    def check_func(self, get_simulation, fiducial):
        if fiducial:
            if len(signature(get_simulation).parameters) != 1:
                print("`get_simulations` must be a function which takes a " +
                      "seed only.")
                sys.exit()
        else:
            if len(signature(get_simulation).parameters) != 3:
                print("`get_simulations` must be a function which takes a " +
                      "seed, derivative and parameter as an argument.")
                sys.exit()

    def check_params(self, n_params, fiducial):
        if not fiducial:
            if n_params is None:
                print("`n_params` must be supplied when making derivative " +
                      "record.")
                sys.exit()

    def _bytes_feature(self, value):
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
