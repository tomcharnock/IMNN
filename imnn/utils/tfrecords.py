import tensorflow as tf
import numpy as np
from inspect import signature
import os
import sys


class TFRecords():
    """ Module for writing simulations to TFRecord to be used by the IMNN

    Parameters
    __________
    record_size : int
        approximate maximum size of an individual record (in Mb)
    padding : int
        zero padding size for record name numbering
    input_shape : tuple
        shape of a single simulation
    file : str
        filename to save the records to
    """
    def __init__(self, record_size=150., padding=5):
        """Constructor method

        Parameters
        ----------
        record_size : float
            The desired size of the final TFRecords (in Mb)
        padding : int
            zero padding size for record name numbering
        """
        self.record_size = record_size * int(1e6)
        self.padding = padding
        self.input_shape = None
        self.file = None

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
        """Serialises a fiducial simulation

        Takes a seed index and a function to get a simulation at that given
        index (and a counter for printing the number of files made). This seed
        and simulation are then serialised and written to file.

        Parameters
        ----------
        seed : int
            index to grab a simulation at
        counter : int
            the number record which is being written to
        get_simulation : fn
            a function which takes an index and returns a simulation
            corresponding to that index

        Returns
        -------
        int:
            the input seed value increased by 1
        """
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
        """Serialises a numerical derivative simulation

        Takes a tuple containing a seed index, an index describing whether the
        corresponding simulation is produced below or above the fiducial
        parameter values and an index describing which parameter the simulation
        is going to be a derivative of. These are used to sequentially get
        simulations from a function until either all of the simulations for the
        derivative of a single seed are collected or until the record is full
        at which it breaks out of the loop to create the next record and
        continue there.

        Parameters
        ----------
        simulation : tuple
            - *(int)* -- index to grab a simulation at
            - *(int)* -- index describing whether generated above or below fid
            - *(int)* -- index for the parameter of interest
        counter : int
            the number record which is being written to
        get_simulation : fn
            a function which takes an index and returns a simulation
            corresponding to that index
        n_params : int
            the number of parameters in the numerical derivative

        Returns
        -------
        tuple
            - *(int)* -- index up to where the simulation was grabbed
            - *(int)* -- index describing whether generated above or below fid
            - *(int)* -- index for the parameter of interest
        """
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
        """Returns the fiducial or derivative serialiser

        Parameters
        ----------
        fiducial: bool
            whether the fiducial serialiser should be returned or not
        get_simulation: fn
            the function which returns either a simulation or a part of a
            derivative simulation
        n_params: int
            the number of parameters that the derivative is taken wrt

        Returns
        -------
        fn:
            either the fiducial serialiser or the derivative serialiser
        """
        if fiducial:
            return lambda inds, counter: self.fiducial_serialiser(
                inds, counter, get_simulation)
        else:
            return lambda inds, counter: self.derivative_serialiser(
                inds, counter, get_simulation, n_params)

    def parser(self, example):
        """Maps a serialised example simulation into its float32 representation

        Parameters
        ----------
        example: str
            The serialised string to be parsed

        Returns
        -------
        float(input_shape):
            The parsed numerical form of the serialised input

        Todo
        ----
        Will fail if input_shape is not set, with no checking. Something should
        be done about this
        """
        parsed_example = tf.io.parse_single_example(
            example, {"data": tf.io.FixedLenFeature([], tf.string)})
        return tf.reshape(
            tf.io.decode_raw(parsed_example["data"], tf.float32),
            self.input_shape)

    def derivative_parser(self, example, n_params=None):
        """Maps a serialised example derivative into its float32 representation

        This is the parser for an analytical or automatic derivative of the
        simulation with respect to model parameters (should be serialised with
        ``fiducial_serialiser``).

        Parameters
        ----------
        example: str
            The serialised string to be parsed
        n_params: int or None, default=None
            The number of parameters in the derivative. This is required but
            named here for use with functools.partial

        Returns
        -------
        float(input_shape, n_params):
            The parsed numerical form of the serialised input

        Todo
        ----
        Will fail if input_shape is not set, with no checking. Something should
        be done about this
        """
        parsed_example = tf.io.parse_single_example(
            example, {"data": tf.io.FixedLenFeature([], tf.string)})
        return tf.reshape(
            tf.io.decode_raw(parsed_example["data"], tf.float32),
            self.input_shape + (n_params,))

    def numerical_derivative_parser(self, example, n_params=None):
        """Maps a serialised example derivative into its float32 representation

        This is the parser for all the simulations necessary for making a
        numerical derivative of a simulation with respect to model parameters
        (should be serialised with ``derivative_serialiser``).

        Parameters
        ----------
        example: str
            The serialised string to be parsed
        n_params: int or None, default=None
            The number of parameters in the derivative. This is required but
            named here for use with functools.partial

        Returns
        -------
        float(2, n_params, input_shape):
            The parsed numerical form of the serialised input

        Todo
        ----
        Will fail if input_shape is not set, with no checking. Something should
        be done about this
        """
        parsed_example = tf.io.parse_single_example(
            example, {"data": tf.io.FixedLenFeature([], tf.string)})
        return tf.reshape(
            tf.io.decode_raw(parsed_example["data"], tf.float32),
            (2, n_params) + self.input_shape)

    def get_file(self, directory, filename, fiducial, validation):
        """Constructs filepath and name that records will be saved to

        Parameters
        ----------
        directory: str
            The full path to where the files should be saved
        filename: str or None
            a filename to save the records to, if None default names are given
            depending on the value of ``fiducial`` and ``validation``
        fiducial: bool
            whether to call a file ``fiducial`` (if ``filename`` is None) or
            ``derivative``
        validation: bool
            whether to prepend ``validation_`` to the filename (if ``filename``
            is None)

        Returns
        -------
        str:
            the filename to save the record to
        """
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
        """Checks the size of the current record in Mb to see whether its full

        Parameters
        ----------
        counter: int
            The current open record being written to

        Returns
        -------
        bool:
            True if the record has reached (exceeded) the preassigned record
            size
        """
        return os.path.getsize(
            ".".join((
                "_".join((
                    self.file,
                    "{}".format(counter).zfill(self.padding))),
                "tfrecords"))) > self.record_size

    def get_initial_seed(self, fiducial, start):
        """Sets the initial seed index (or set of seeds) to get simulations at

        When constructing a record with a fiducial (or exact derivative) only
        a seed index for the simulation is needed whilst a derivative index and
        a parameter index are also needed if a numerical derivative simulation
        is being collected. Seed indexs will increase incrementally by 1.

        Parameters
        ----------
        fiducial: bool
            if a fiducial simulation record is being constructed
        start: int
            the initial seed index to collect the simulation at

        Returns
        -------
        int or tuple:
            - *(int)* -- the initial seed index to collect the simulations at
            - *(tuple)*
                - *(int)* -- the initial seed index to collect the simulation
                - *(int)* -- whether the simulation is generated below or
                  above the fiducial parameter values
                - *(int)* -- which respect to which parameter the simulation is
                  used to calculate the numerical gradient
        """
        if fiducial:
            return start
        else:
            return (start, 0, 0)

    def get_seed(self, simulation, fiducial):
        """Gets the seed index depending on input

        With a fiducial (or exact derivative) only a seed index for the
        simulation is needed whilst a derivative index and a parameter index
        are also needed if a numerical derivative simulation is being collected

        Parameters
        ----------
        simulation: tuple
            - *(int)* -- the initial seed index to collect the simulation
            - *(int)* -- whether the simulation is generated below or above the
              fiducial parameter values
            - *(int)* -- which respect to which parameter the simulation is
              used to calculate the numerical gradient
        fiducial: bool
            if a fiducial simulation record is being constructed

        Returns
        -------
        int
            the seed index to collect the simulation at
        """
        if fiducial:
            return simulation
        else:
            return simulation[0]

    def check_func(self, get_simulation, fiducial):
        """Checks the simulation grabber takes the correct number of arguments

        Parameters
        ----------
        get_simulation : fn
            a function to get a single simulation
        fiducial :
            if a fiducial simulation record is being constructed

        Todo
        ----
        Should really make these raise ValueErrors rather than using sys to
        shut down
        """
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
        """Checks that n_params is actually given

        Parameters
        ----------
        n_params : int
            the number of parameters in the derivative
        fiducial :
            if a fiducial simulation record is being constructed
        """
        if not fiducial:
            if n_params is None:
                print("`n_params` must be supplied when making derivative " +
                      "record.")
                sys.exit()

    def _bytes_feature(self, value):
        """Makes a serialised byte list from value (converts tensors to numpy)

        Parameters
        ----------
        value : float (possibly Tensor)
            the simulation to be serialised

        Returns
        -------
        byte_list:
            the simulation as a string of bytes
        """
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(self, value):
        """Makes a serialised int list from value

        Parameters
        ----------
        value : int
            the seed to be serialised

        Returns
        -------
        int_list:
            the serialised int list
        """
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
