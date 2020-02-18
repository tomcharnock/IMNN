"""Information maximising neural network
This module provides the methods necessary to build and train an information
maximising neural network to optimally compress data down to the number of
model parameters.

TODO
____
Still some docstrings which need finishing
Use precomputed external covariance and derivatives"""


__version__ = '0.2a4'
__author__ = "Tom Charnock"


import tensorflow as tf
import numpy as np
import tqdm
from IMNN.utils import utils


class IMNN():
    """Information maximising neural network
    The information maximising neural network class contains all the functions
    necessary to train a neural network to maximise the Fisher information of
    the summaries of the input data.
    Attributes
    __________
    u : class
        model of functions for parameter error checking
    dtype : TF type
        32 bit or 64 TensorFlow tensor floats
    itype : TF type
        32 bit or 64 TensorFlow tensor integers
    save : bool
        whether to save the model
    filename : str
        directory to save the model
    validate : bool
        whether to validate during training
    n_s : int
        number of simulations to calculate summary covariance
    n_d : int
        number of derivatives simulations to calculate derivative of mean
    n_params : int
        number of parameters in physical model
    n_summaries : int
        number of summaries to compress data to
    model : TF model - keras or other
        neural network to do the compression defined using TF or keras
    optimiser : TF optimiser - keras or other
        optimisation operation to do weight updates, defined using TF or keras
    θ_fid : TF tensor float (n_params,)
        fiducial parameter values for training dataset
    δθ : TF tensor float (n_params,)
        parameter differences for numerical derivatives
    input_shape : tuple
        shape of the input data
    data : TF tensor float (n_s,) + input_shape
        fiducial data for training IMNN
    derivative : TF tensor float (n_d, 2, n_params) + input_shape
        derivatives of the data for training IMNN (second index 0=-ve, 1=+ve)
    validation_data : TF tensor float (n_s,) + input_shape
        fiducial data for validating IMNN
    validation_derivative : TF tensor float (n_d, 2, n_params) + input_shape
        derivatives of the data for validating IMNN (second index 0=-ve, 1=+ve)
    fiducial_at_once : int
        number of simulations to process at once if using TF.data.Dataset
    derivatives_at_once : int
        number of simulations to process at once if using TF.data.Dataset
    fiducial_dataset : TF dataset
        dataset to grab large fiducial data for training IMNN
    derivative_dataset : TF dataset
        dataset to grab large derivative data for training IMNN
    validation_fiducial_dataset : TF dataset
        dataset to grab large fiducial data for validating IMNN
    validation_derivative_dataset : TF dataset
        dataset to grab large derivative data for validating IMNN
    F : TF tensor float (n_params, n_params)
        Fisher information matrix
    C : TF tensor float (n_summaries, n_summaries)
        covariance of summaries
    Cinv : TF tensor float (n_summaries, n_summaries)
        inverse covariance of summaries
    μ : TF tensor float (n_summaries)
        mean of the summaries
    dμ_dθ : TF tensor float (n_params, n_summaries)
        derivative of mean summaries with respect to the parameters
    reg : TF tensor float ()
        value of regulariser
    r : TF tensor float ()
        value of the dynamical coupling strength of the regulariser
    λ : TF tensor float ()
        coupling strength for the regularisation
    α : TF tensor float ()
        decay rate for the regularisation
    history : dict
        history object for saving training statistics.
    """
    def __init__(self, n_s, n_d, n_params, n_summaries, model, optimiser,
                 θ_fid, δθ, fiducial_loader, derivative_loader,
                 dtype=tf.float32, itype=tf.int32, save=False, verbose=True,
                 filename=None, at_once=None, validation_fiducial_loader=None,
                 validation_derivative_loader=None, map_fn=None,
                 check_shape=True):
        """Initialises attributes and calculates useful constants

        Parameters
        __________
        n_s : int
            number of simulations to calculate summary covariance
        n_d : int
            number of derivatives simulations to calculate derivative of mean
        n_params : int
            number of parameters in physical model
        n_summaries : int
            number of summaries to compress data to
        model : TF model (keras or other)
            neural network to do the compression defined using TF or keras
        optimiser : TF optimiser (keras or other)
            optimisation operation to do weight updates using TF or keras
        θ_fid : ndarray (n_params,)
            fiducial parameter values for training dataset
        δθ : ndarray (n_params,)
            parameter differences for numerical derivatives
        at_once : int
            number of simulations to process at once if using TF.data.Dataset
        fiducial_loader : ndarray (n_s,) + input_shape or func
            numpy array containing fiducial data or function returning same
        derivative_loader : ndarray (n_d, 2, n_params) + input_shape or func
            numpy array containing derivative data or function returning same
        validation_fiducial_loader : nd_array (n_s,) + input_shape or func
            numpy array containing fiducial data or function returning same
        validation_derivative_loader : nd_array or func
            numpy array containing derivative data or function returning same
        map_fn : func
            function for data augmentation when using TF datasets
        dtype : TF type
            tensorflow double size
        itype : TF type
            tensorflow interger size
        save : bool
            whether to save the model
        filename : str
            name for saving the model
        verbose : bool
            whether to use verbose outputs in error checking module
        check_shape : bool
            whether to check the shape of the model and the data

        Calls
        _____
        IMNN.utils.utils.type_checking(any, type, str, opt(str))
            checks that value exists and is of the correct type
        init_attributes(int, int, int, int, tfdtype, tfdtype, bool, str, bool)
            Initialises all attributes and sets necessary constants
        set_tensors(ndarray, ndarray)
            sets useful tensors which can be precomputed
        set_data(ndarray/gen, ndarray/gen, ndarray/gen, ndarray/gen,
                 int, func, bool)
            builds the datasets and sets the functions to be used to train IMNN
        set_model(model, optimiser)
            loads the model and optimiser as attributes
        """
        self.u = utils.utils(verbose=verbose)
        check_shape = self.u.type_checking(check_shape, True, "check_shape")
        self.init_attributes(n_s, n_d, n_params, n_summaries, dtype, itype,
                             save, filename, verbose)
        self.set_tensors(θ_fid, δθ)
        self.set_data(fiducial_loader, derivative_loader,
                      validation_fiducial_loader, validation_derivative_loader,
                      at_once, map_fn, check_shape)
        self.set_model(model, optimiser)

    def init_attributes(self, n_s, n_d, n_params, n_summaries,
                        dtype, itype, save, filename, verbose):
        """Initialises all attributes and sets necessary constants

        All attributes are set to None before they are loaded when
        necessary. The number of parameters and summaries are set and
        the number of simulations needed for the covariance and derivatives
        of the mean summaries is set.

        Parameters
        __________
        n_s : int
            number of simulations to calculate summary covariance
        n_d : int
            number of derivatives simulations to calculate derivative of mean
        n_params : int
            number of parameters in physical model
        n_summaries : int
            number of summaries to compress data to
        dtype : TF type
            tensorflow double size
        itype : TF type
            tensorflow interger size
        save : bool
            whether to save the model
        filename : str
            name for saving the model
        verbose : bool
            whether to use verbose outputs in error checking module

        Calls
        _____
        initialise_history()
            returns dictionary of lists to be populated during training
        IMNN.utils.utils.type_checking(any, type, str, opt(str))
            checks that value exists and is of the correct type
        IMNN.utils.utils.positive_integer(int, str) -> int
            checks whether parameter is positive integer and error otherwise
        """
        self.verbose = self.u.type_checking(verbose, True, "verbose")

        self.dtype = self.u.type_checking(dtype,
                                          tf.float32,
                                          "dtype")
        self.itype = self.u.type_checking(itype,
                                          tf.int32,
                                          "itype")

        self.save = self.u.type_checking(save, True, "save")
        if self.save:
            if filename is None:
                self.filename = "model"
            else:
                self.filename = self.u.type_checking(filename,
                                                     "hello",
                                                     "filename")
        else:
            self.filename = None

        self.n_s = self.u.positive_integer(n_s, "n_s")
        self.n_d = self.u.positive_integer(n_d, "n_d")
        self.n_params = self.u.positive_integer(n_params, "n_params")
        self.n_summaries = self.u.positive_integer(n_summaries, "n_summaries")

        self.history = self.initialise_history()

        self.model = None
        self.optimiser = None

        self.θ_fid = None
        self.δθ = None

        self.data = None
        self.derivative = None
        self.fiducial_dataset = None
        self.derivative_dataset = None
        self.fiducial_at_once = None
        self.derivative_at_once = None

        self.validate = None
        self.validation_data = None
        self.validation_derivative = None
        self.validation_fiducial_dataset = None
        self.validation_derivative_dataset = None

        self.F = None
        self.Finv = None
        self.C = None
        self.Cinv = None
        self.μ = None
        self.dμ_dθ = None
        self.reg = None
        self.r = None
        self.λ = None
        self.α = None
        self.identity = None

    def initialise_history(self):
        """Sets up dictionary of lists for collecting training diagnostics

        Dictionary of all diagnostics which can be collected during training.
        These are:
            det_F - determinant of Fisher information
            val_det_F - determinant of Fisher information from validation
            det_C - determinant of covariance of summaries
            val_det_C - determinant of covariance of validation summaries
            det_Cinv - determinant of inverse covariance of summaries
            val_det_Cinv - det of inverse covariance of validation summaries
            dμ_dθ - derivative of mean summaries wrt model parameters
            val_dμ_dθ - derivative of mean validation summaries wrt parameters
            reg - value of the regularisation term
            r - value of the coupling strength of the regulariser
        """
        return {
            "det_F": [],
            "val_det_F": [],
            "det_C": [],
            "val_det_C": [],
            "det_Cinv": [],
            "val_det_Cinv": [],
            "dμ_dθ": [],
            "val_dμ_dθ": [],
            "reg": [],
            "r": [],
        }

    def set_model(self, model, optimiser):
        """Loads functional neural network and optimiser as attributes

        Parameters
        __________
        model : TF model (keras or other)
            neural network to do the compression defined using TF or keras
        optimiser : TF optimiser (keras or other)
            optimisation operation to do weight updates using TF or keras

        Calls
        _____
        check_model(model, tuple, int)
            checks that model takes expected input shape and output n_summaries
        """
        self.model = self.u.check_model(model,
                                        self.input_shape,
                                        self.n_summaries)
        self.optimiser = optimiser
        if self.save:
            if self.verbose:
                print("saving model to " + self.filename)
            self.model.save(self.filename)

    def load_model(self, optimiser, weights=None):
        """Reloads a saved model

        Parameters
        __________
        optimiser : TF optimiser (keras or other)
            optimisation operation to do weight updates using TF or keras
        weights : str
            filename for saving weights
        """
        self.model = tf.keras.models.load_model(self.filename)
        self.optimiser = optimiser
        if weights is not None:
            self.model.load_weights(self.filename + "/" + weights + ".h5")

    def set_tensors(self, θ_fid, δθ):
        """Makes TF tensors for necessary objects which can be precomputed

        Sets up the loop variable tensors for training and validation and
        calculates the derivative of the mean summaries with respect to outputs
        which can be precomputed. Also makes tensor float version of number of
        simulations for summary covariance and this value minus 1 for unbiased
        covariance calculation. The identity matrix is also defined.

        Parameters
        __________
        θ_fid : ndarray (n_params,)
            fiducial parameter values for training dataset
        δθ : ndarray (n_params,)
            parameter differences for numerical derivatives

        Calls
        _____
        IMNN.utils.utils.check_shape(any, type, tuple, str, opt(str))
            checks that value is of the correct type and shape
        """
        self.F = tf.zeros((self.n_params, self.n_params),
                          dtype=self.dtype,
                          name="fisher")
        self.Finv = tf.zeros((self.n_params, self.n_params),
                             dtype=self.dtype,
                             name="inverse_fisher")
        self.C = tf.zeros((self.n_summaries, self.n_summaries),
                          dtype=self.dtype,
                          name="summary_covariance")
        self.Cinv = tf.zeros((self.n_summaries, self.n_summaries),
                             dtype=self.dtype,
                             name="inverse_summary_covariance")
        self.μ = tf.zeros((self.n_summaries,),
                          dtype=self.dtype,
                          name="summary_mean")
        self.dμ_dθ = tf.zeros((self.n_params, self.n_summaries),
                              dtype=self.dtype,
                              name="mean_summary_derivative")
        self.reg = tf.zeros((), dtype=self.dtype, name="regularisation")
        self.r = tf.zeros((), dtype=self.dtype, name="r")

        self.identity = tf.eye(self.n_summaries, name="summary_identity")
        self.θ_fid = tf.convert_to_tensor(
            self.u.check_shape(
                θ_fid, np.zeros(()), (self.n_params,), "θ_fid"),
            dtype=self.dtype,
            name="fiducial_parameters")
        self.δθ = tf.convert_to_tensor(
            self.u.check_shape(δθ,
                               np.zeros(()),
                               (self.n_params,),
                               "δθ")[np.newaxis, :, np.newaxis],
            dtype=self.dtype,
            name="derivative_steps")

    def set_data(self, fiducial_loader, derivative_loader,
                 validation_fiducial_loader, validation_derivative_loader,
                 at_once, map_fn, check_shape):
        """Builds the datasets and sets the functions to be used to train IMNN

        fiducial_loader and derivative_loader must both be either np.ndarrays
        with shape (n_s,) + input_shape and (n_d, 2, n_params) + input_shape
        respectively, or generators which take a single index and yield the
        data for the fiducial data and the value of the index and for the
        dereivatives the generator takes a simulation index, a derivative
        index which is 0 for the lower simulations and 1 for the upper
        simulations and a parameter index labelling which parameter is selected
        and returns the data and a tuple of the three indices.

        Loaders can be passed for the validation data. It is strictly necessary
        to have the same type of loaders for training and validation (both
        ndarray or both generators).

        If data augmentation is part of the pipeline this can be performed on
        the data by passing a map_fn which is a function that takes in one
        piece of data and augments it and returns just the one output.

        Parameters
        __________
        fiducial_loader : ndarray (n_s,) + input_shape or func
            numpy array containing fiducial data or function returning same
        derivative_loader : ndarray (n_d, 2, n_params) + input_shape or func
            numpy array containing derivative data or function returning same
        validation_fiducial_loader : nd_array (n_s,) + input_shape or func
            numpy array containing fiducial data or function returning same
        validation_derivative_loader : nd_array or func
            numpy array containing derivative data or function returning same
        at_once : int
            number of simulations to process at once
        map_fn : func
            function for data augmentation when using TF datasets
        check_shape : bool
            whether to check the shape of the model and the data

        Calls
        _____
        IMNN.utils.utils.check_shape(any, type, tuple, str, opt(str))
            checks that value is of the correct type and shape
        IMNN.utils.utils.data_error(opt(bool))
            prints warning if data is not correct
        build_dataset(func, bool, opt(func))
            builder for the tf.data.Dataset based on loading function
        """
        if ((type(fiducial_loader) is np.ndarray) and
                (type(derivative_loader) is np.ndarray)):
            self.input_shape = fiducial_loader.shape[1:]
            if self.verbose:
                print("input_shape = " + str(self.input_shape)
                      + ". If this is not what you expected, check your data.")
            if check_shape:
                fiducial_loader = self.u.check_shape(
                    fiducial_loader,
                    np.zeros(()),
                    (self.n_s,) + self.input_shape,
                    "fiducial_loader")
                derivative_loader = self.u.check_shape(
                    derivative_loader,
                    np.zeros(()),
                    (self.n_d, 2, self.n_params) + self.input_shape,
                    "derivative_loader")
            self.data = tf.convert_to_tensor(fiducial_loader,
                                             dtype=self.dtype)
            self.derivative = tf.convert_to_tensor(derivative_loader,
                                                   dtype=self.dtype)
            fast = True
            self.trainer = self.fast_train
        elif callable(fiducial_loader) and callable(derivative_loader):
            self.fiducial_at_once, self.derivative_at_once = \
                self.u.at_once_checker(at_once, self.n_s, self.n_d,
                                       self.n_params)
            temp_data = next(fiducial_loader(0))[0]
            self.input_shape = temp_data.shape
            del(temp_data)
            if self.verbose:
                print("input_shape = " + str(self.input_shape)
                      + ". If this is not what you expected, check your data.")
            self.fiducial_dataset = self.build_dataset(fiducial_loader,
                                                       derivative=False,
                                                       map_fn=map_fn)
            self.derivative_dataset = self.build_dataset(derivative_loader,
                                                         derivative=True,
                                                         map_fn=map_fn)
            fast = False
            self.trainer = self.scatter
        else:
            self.u.data_error()

        if ((validation_fiducial_loader is not None)
                and (validation_derivative_loader is not None)):
            self.validate = True
            if ((fast) and
                    (type(validation_fiducial_loader) is np.ndarray) and
                    (type(validation_derivative_loader) is np.ndarray)):
                if check_shape:
                    validation_fiducial_loader = self.u.check_shape(
                        validation_fiducial_loader,
                        np.zeros(()),
                        (self.n_s,) + self.input_shape,
                        "validation_fiducial_loader")
                    validation_derivative_loader = self.u.check_shape(
                        validation_derivative_loader,
                        np.zeros(()),
                        (self.n_d, 2, self.n_params) + self.input_shape,
                        "validation_derivative_loader")
                self.validation_data = tf.convert_to_tensor(
                    validation_fiducial_loader,
                    dtype=self.dtype)
                self.validation_derivative = tf.convert_to_tensor(
                    validation_derivative_loader,
                    dtype=self.dtype)
                self.validater = self.run_fisher
            elif ((not fast) and
                    callable(validation_fiducial_loader) and
                    callable(validation_derivative_loader)):
                self.validation_fiducial_dataset = self.build_dataset(
                    validation_fiducial_loader,
                    derivative=False,
                    map_fn=map_fn)
                self.validation_derivative_dataset = self.build_dataset(
                    validation_derivative_loader,
                    derivative=True,
                    map_fn=map_fn)
                self.validater = self.scatter
            else:
                self.u.data_error(validate=True)
        elif ((validation_fiducial_loader is None) and
              (validation_derivative_loader is None)):
            self.validate = False
        else:
            self.u.data_error(validation=True)

    def build_dataset(self, loader, derivative, map_fn):
        """Build tf.data.Dataset for the necessary datasets

        Parameters
        __________
        loader : generator
            a generator which returns the data and indices used to select it
        derivative : bool
            whether the dataset is for the fiducial or derivative data
        map_fn : func
            a function taking a datum and augmenting it as part of the pipeline
        """
        if derivative:
            size = self.n_d
            at_once = self.derivative_at_once
        else:
            size = self.n_s
            at_once = self.fiducial_at_once

        dataset = tf.data.Dataset.range(size)
        dataset = dataset.map(lambda x: tf.cast(x, dtype=self.itype))

        if derivative:
            dataset = dataset.map(
                lambda x: (
                    tf.tile(
                        tf.expand_dims(x, 0),
                        tf.expand_dims(self.n_params * 2, 0)),
                    tf.tile(
                        tf.range(2, dtype=self.itype),
                        tf.expand_dims(self.n_params, 0)),
                    tf.repeat(
                        tf.range(self.n_params, dtype=self.itype),
                        2)))
            dataset = dataset.unbatch()
            dataset = dataset.interleave(
                lambda x, y, z: tf.data.Dataset.from_generator(
                    loader,
                    (self.dtype, (self.itype, self.itype, self.itype)),
                    (tf.TensorShape(self.input_shape), (tf.TensorShape(None),
                     tf.TensorShape(None), tf.TensorShape(None))),
                    args=(x, y, z)),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        else:
            dataset = dataset.interleave(
                lambda x: tf.data.Dataset.from_generator(
                    loader,
                    (self.dtype, self.itype),
                    (tf.TensorShape(self.input_shape), tf.TensorShape(None)),
                    args=(x,)),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if map_fn is not None:
            dataset = dataset.map(
                lambda data, indices:
                    (map_fn(data), indices))
        dataset = dataset.batch(at_once)
        return dataset.prefetch(tf.data.experimental.AUTOTUNE)

    def set_dataset(self, derivative, validate):
        """Grabs the necessary dataset and the function to create indices

        Parameters
        __________
        derivative : bool
            whether the dataset is for derivatives or not
        validate : bool
            whether the dataset is for validation or training

        Returns
        _______
        dataset : TF dataset
            fiducial or derivative dataset (for training or validation)
        get_indices : func
            function for creating mesh of indices for fiducial or derivatives
        """
        if derivative:
            if validate:
                return self.validation_derivative_dataset, self.get_derivative_indices
            else:
                return self.derivative_dataset, self.get_derivative_indices
        else:
            if validate:
                return self.validation_fiducial_dataset, self.get_fiducial_indices
            else:
                return self.fiducial_dataset, self.get_fiducial_indices

    def get_fiducial_indices(self, index):
        """Constructs the mesh of indices for scattering fiducial summaries

        Parameters
        __________
        index : TF tensor int (batch,)
            the indices of the data to aggregate summaries

        Returns
        _______
            indices : TF tensor int (batch, n_summaries)
                mesh of indices for scattering fiducial summaries
        """
        return tf.stack(
            tf.meshgrid(
                index,
                tf.range(self.n_summaries, dtype=self.itype),
                indexing="ij"),
            axis=-1)

    def get_derivative_indices(self, index):
        """Constructs the mesh of indices for scattering derivative summaries

        This is performed as a vectorised mapping of meshgrid.

        Parameters
        __________
        index : tuple(tf.int, tf.int, tf.int) (batch, 3)
            the indices of the data to aggregate summaries

        Returns
        _______
        indices : TF tensor int (batch, n_summaries, 4)
            mesh of indices for scattering derivative summaries
        """
        return tf.vectorized_map(
            lambda i: tf.squeeze(
                tf.stack(
                    tf.meshgrid(
                        i[0],
                        i[1],
                        i[2],
                        tf.range(self.n_summaries, dtype=self.itype),
                        indexing="ij"),
                    axis=-1),
                (0, 1, 2)),
            index)

    def get_summaries(self, x, derivative=False, validate=False):
        """Passes the data through the model and aggregates it to a tensor

        Parameters
        __________
        x : TF tensor float (n_s, n_summaries)/(n_d, 2, n_params, n_summaries)
            summaries outputted from the network
        derivative : bool
            whether the dataset is for derivatives or not
        validate : bool
            whether the dataset is for validation or training

        Returns
        _______
        x : TF tensor float (n_s, n_summaries)/(n_d, 2, n_params, n_summaries)
            summaries outputted from the network

        Calls
        _____
        set_dataset(bool, bool)
            returns the necessary dataset and the function to create indices
        get_fiducial_indices(tf.int)
            returns the mesh of indices for scattering fiducial summaries
        get_derivative_indices((tf.int, tf.int, tf.int))
            returns the mesh of indices for scattering derivative summaries
        """
        dataset, get_indices = self.set_dataset(derivative, validate)
        for data, index in dataset:
            indices = get_indices(index)
            x = tf.tensor_scatter_nd_update(
                x,
                indices,
                self.model(data))
        return x

    def get_covariance(self, x):
        """Calculates covariance, mean and difference of mean from summaries

        Calculates the mean of the summaries and then finds the difference
        between the mean and each summary. This can then be used to calculate
        the covariance of the summaries.

        Parameters
        __________
        x : TF tensor float (n_s, n_summaries)
            summaries output by the model

        Returns
        _______
        C : TF tensor float (n_summaries, n_summaries)
            covariance of the summaries
        Cinv : TF tensor float (n_summaries, n_summaries)
            inverse covariance of the summaries
        μ : TF tensor float (n_summaries,)
            mean of the summaries
        """
        μ = tf.reduce_mean(x, axis=0, keepdims=True)
        ν = tf.subtract(x, μ)
        C = tf.divide(
            tf.einsum("ij,ik->jk", ν, ν),
            tf.convert_to_tensor(self.n_s - 1., dtype=self.dtype))
        Cinv = tf.linalg.inv(C)
        return C, Cinv, tf.squeeze(μ, 0)

    def get_dμ_dθ(self, dx_dθ):
        """ Calculates the mean of the derivative of the summaries

        Numerically, we take away the simulations generated above the fiducial
        parameter values from the simulations generated below the fiducial
        parameter values and divide them by the difference between the
        upper and lower parameter values.

        Parameters
        _________
        dx_dθ : TF tensor float (n_d, 2, n_params, n_summaries)
            upper and lower parameter value summaries for numerical derivative

        Returns
        _______
        dμ_dθ : TF tensor float (n_params, n_summaries)
            derivative of the mean of the summaries with respect to the params
        """
        return tf.reduce_mean(
            tf.divide(
                tf.subtract(
                    dx_dθ[:, 1, :, :],
                    dx_dθ[:, 0, :, :]),
                self.δθ),
            axis=0)

    def get_fisher(self, Cinv, dμ_dθ):
        """Calculate Fisher information matrix

        Parameters
        __________
        Cinv : TF tensor float (n_summaries, n_summaries) or
                          (n_summaries + n_external, n_summaries + n_external)
            inverse covariance of the summaries
        dμ_dθ : TF tensor float (n_params, n_summaries) or
                                (n_params, n_summaries + n_external)
            derivative mean summaries wrt parameters

        Returns
        _______
        F : TF tensor float (n_params, n_params)
            Fisher information matrix
        """
        return tf.einsum(
            "ij,kj->ik",
            dμ_dθ,
            tf.einsum(
                "ij,kj->ki",
                Cinv,
                dμ_dθ))

    def get_regularisation(self, C, Cinv):
        """Calculate the regulariser to set scale of the summaries

        The Frobenius norm of the difference between the covariance and the
        inverse covariance from the identity is calculated to set the scale of
        the summaries outputted by the IMNN.

        Parameters
        __________
        C : TF tensor float (n_summaries, n_summaries)
            covariance of the summaries
        Cinv : TF tensor float (n_summaries, n_summaries)
            inverse covariance of the summaries

        Returns
        _______
        regularisation : TF tensor float ()
            value of the regularisation term
        """
        return tf.multiply(
            tf.convert_to_tensor(0.5, dtype=self.dtype),
            tf.add(
                tf.square(
                    tf.norm(tf.subtract(C, self.identity),
                            ord="fro",
                            axis=(0, 1))),
                tf.square(
                    tf.norm(tf.subtract(Cinv, self.identity),
                            ord="fro",
                            axis=(0, 1)))))

    def get_r(self, regularisation):
        """Dynamical coupling strength for the regularisation

        To make sure there is a smooth surface which turns off the
        regularisation, we use a dynamical coupling strength which goes from
        zero when the covariance is identity to some strength λ when the
        covariance is away from identity. The rate of this is controlled by how
        close to the identity we want the covariance to be.

        Parameters
        __________
        regularisation : TF tensor float ()
            value of the regularisation term

        Returns
        _______
        r : TF tensor float ()
            dynamical coupling strength for the regularisation
        """
        return tf.divide(
            tf.multiply(self.λ, regularisation),
            tf.add(
                regularisation,
                tf.exp(tf.multiply(-self.α, regularisation))))

    def get_loss(self, F, regularisation, r):
        """Calculate the loss function (-log(det(F)) + regularisation)

        Parameters
        __________
        F : TF tensor float (n_params, n_params)
            Fisher information matrix
        regularisation : TF tensor float ()
            Frobenium norm of difference between C and I and Cinv and I
        r : TF tensor float ()
            dynamic regulariser coupling strength determined by regularisation

        Returns
        _______
        Λ : TF tensor float ()
            -log(det(F)) + regularisation
        """
        return tf.subtract(
            tf.multiply(r, regularisation),
            tf.linalg.slogdet(F))

    def calculate_fisher(self, x, dx_dθ):
        """Calculate necessary stats to get fisher information

        Parameters
        __________
        x : TF tensor float (n_s, n_summaries)
            summaries output by the model
        dx_dθ : TF tensor float (n_d, 2, n_params, n_summaries)
            upper and lower parameter value summaries for numerical derivative

        Returns
        _______
        F : TF tensor float (n_params, n_params)
            Fisher information matrix
        C : TF tensor float (n_summaries, n_summaries)
            covariance of summaries
        Cinv : TF tensor float (n_summaries, n_summaries)
            inverse covariance of summaries
        μ : TF tensor float (n_summaries)
            mean of the summaries
        dμ_dθ : TF tensor float (n_params, n_summaries)
            derivative of mean summaries with respect to the parameters

        Calls
        _____
        get_covariance(tensor)
            calculates covariance, mean and difference of mean from summaries
        get_dμ_dθ(tensor)
            calculates the mean of the derivative of the summaries
        get_fisher(tensor)
            calculates Fisher information matrix
        """
        C, Cinv, μ = self.get_covariance(x)
        dμ_dθ = self.get_dμ_dθ(dx_dθ)
        F = self.get_fisher(Cinv, dμ_dθ)
        return F, C, Cinv, μ, dμ_dθ

    def calculate_loss(self, F, C, Cinv):
        """Calculate the regularisation and loss function

        Parameters
        __________
        F : TF tensor float (n_params, n_params)
            Fisher information matrix
        C : TF tensor float (n_summaries, n_summaries)
            covariance of summaries
        Cinv : TF tensor float (n_summaries, n_summaries)
            inverse covariance of summaries

        Returns
        _______
        Λ : TF tensor float ()
            value of the regularised -log(det(F))
        regularisation : TF tensor float ()
            value of the regularisation term
        r : TF tensor float ()
            value of the dynamical regularisation coupling strength

        Calls
        _____
        get_regularisation(tensor, tensor)
            calculates the regulariser to set scale of the summaries
        get_r(tensor)
            calculates the dynamical coupling strength for the regularisation
        get_loss(tensor, tensor, tensor)
            calculates the loss function (-log(det(F)) + regularisation)
        """
        regularisation = self.get_regularisation(C, Cinv)
        r = self.get_r(regularisation)
        Λ = self.get_loss(F, regularisation, r)
        return Λ, regularisation, r

    def get_fisher_gradient(self, x, dx_dθ):
        """Calculates the gradient of the loss with respect to the summaries

        From the summaries, the Fisher information matrix and loss function can
        be calculated, and the gradient taken and returned

        Parameters
        __________
        x : tensor (n_s, n_summaries)
            summaries output by the network
        dx_dθ : tensor (n_d, 2, n_params, n_summaries)

        Returns
        _______
        F : TF tensor float (n_params, n_params)
            Fisher information matrix
        dΛdx : list of TF tensor float (len = 2)
            list of the gradients with respect to the summaries and derivatives
        C : TF tensor float (n_summaries, n_summaries)
            covariance of summaries
        Cinv : TF tensor float (n_summaries, n_summaries)
            inverse covariance of summaries
        μ : TF tensor float (n_summaries)
            mean of the summaries
        dμ_dθ : TF tensor float (n_params, n_summaries)
            derivative of mean summaries with respect to the parameters
        regularisation : TF tensor float ()
            value of regulariser
        r : TF tensor float ()
            value of the dynamical coupling strength of the regulariser

        Calls
        _____
        calculate_fisher(tensor, tensor)
            calculates the statistics necessary for Fisher from the summaries
        calculate_loss(tensor, tensor, tensor)
            calculates the loss function and the regularisation
        """
        with tf.GradientTape() as tape:
            tape.watch([x, dx_dθ])
            F, C, Cinv, μ, dμ_dθ = self.calculate_fisher(x, dx_dθ)
            Λ, regularisation, r = self.calculate_loss(F, C, Cinv)
        dΛdx = tape.gradient(Λ, [x, dx_dθ])
        return F, dΛdx, C, Cinv, μ, dμ_dθ, regularisation, r

    def get_network_gradient(self, dΛdx, derivative=False):
        """Calculates aggregated gradient of the loss with respect to the model

        Using the gradient of the loss with respect to the model we collect and
        aggregate the gradient of the loss with respect to the model parameters
        by passing the data through the model in batches for a second time.

        Parameters
        __________
        dΛdx : tensor float (n_s, n_summaries)/(n_d, 2, n_params, n_summaries)
            derivative of the loss function with respect to the summaries
        derivative : bool
            whether or not the derivative gradient is being calculated

        Returns
        _______
        gradient : list of tensor (len(model.variables))
            gradients of the loss function with respect to model parameters

        Calls
        _____
        set_dataset(bool, bool)
            returns dataset and index making function for training/validation
        """
        dataset, get_indices = self.set_dataset(derivative, validate=False)
        gradient = tuple(tf.zeros(variable.shape)
                         for variable in self.model.variables)
        for data, index in dataset:
            indices = get_indices(index)
            with tf.GradientTape() as tape:
                x = self.model(data)
            batch_gradient = tape.gradient(
                x,
                self.model.variables,
                output_gradients=tf.gather_nd(dΛdx, indices))
            gradient = tuple(
                tf.add(batch_gradient[variable], gradient[variable])
                for variable in range(len(self.model.variables)))
        return gradient

    @tf.function
    def scatter(self, F, C, Cinv, μ, dμ_dθ, reg=None, r=None, validate=False):
        """Scattered calculation of Fisher information and gradients

        The Fisher information is maximised by automatically calculating the
        derivative of the logarithm of the determinant of the Fisher matrix
        regularised by the Frobenius norm of the elementwise difference of the
        summary covariance and the inverse covariance of the summaries from the
        identity matrix. The regulariser is necessary to set the scale of the
        summaries, i.e. the set of summaries which are preferably orthogonal
        with covariance of I. The strength of the regularisation is smoothly
        reduced as the summaries approach a covariance of the identity matrix
        so that preference is given to optimising the Fisher information
        matrix.

        Batches of data are passed through the model and the summaries are
        collected and the statistics are then calculated via these scattered
        tensors. If scatter is being used to train the model then the data is
        passed through the model a second time to calculate aggregated
        gradients. This method is somewhat slower than running directly with
        tensors, but is necssary for large data

        Parameters
        __________
        F : TF tensor float (n_params, n_params)
            Fisher information matrix
        C : TF tensor float (n_summaries, n_summaries)
            covariance of summaries
        Cinv : TF tensor float (n_summaries, n_summaries)
            inverse covariance of summaries
        μ : TF tensor float (n_summaries)
            mean of the summaries
        dμ_dθ : TF tensor float (n_params, n_summaries)
            derivative of mean summaries with respect to the parameters
        reg : TF tensor float ()
            value of regulariser
        r : TF tensor float ()
            value of the dynamical coupling strength of the regulariser
        validate : bool
            whether the data is validation data or not

        Returns
        _______
        F : TF tensor float (n_params, n_params)
            Fisher information matrix
        C : TF tensor float (n_summaries, n_summaries)
            covariance of summaries
        Cinv : TF tensor float (n_summaries, n_summaries)
            inverse covariance of summaries
        μ : TF tensor float (n_summaries)
            mean of the summaries
        dμ_dθ : TF tensor float (n_params, n_summaries)
            derivative of mean summaries with respect to the parameters
        reg : TF tensor float ()
            value of regulariser
        r : TF tensor float ()
            value of the dynamical coupling strength of the regulariser

        Calls
        _____
        get_summaries(tensor, bool, bool)
            passes the data through the model and aggregates the summaries
        calculate_fisher(tensor, tensor)
            calculates the statistics necessary for Fisher from the summaries
        get_fisher_gradient(tensor, tensor)
            calculates the gradient of the loss with respect to the summaries
        get_network_gradient(tensor, bool)
            gets aggregated derivative of the loss with respect to the model
        """
        x = self.get_summaries(tf.zeros((self.n_s, self.n_summaries)),
                               derivative=False,
                               validate=validate)
        dx_dθ = self.get_summaries(
            tf.zeros((self.n_d, 2, self.n_params, self.n_summaries)),
            derivative=True,
            validate=validate)
        if validate:
            return self.calculate_fisher(x, dx_dθ)
        else:
            t = self.get_fisher_gradient(x, dx_dθ)
            fiducial_gradient = self.get_network_gradient(
                t[1][0],
                derivative=False)
            derivative_gradient = self.get_network_gradient(
                t[1][1],
                derivative=True)
            self.optimiser.apply_gradients(
                zip([tf.add(
                        fiducial_gradient[variable],
                        derivative_gradient[variable])
                     for variable in range(len(self.model.variables))],
                    self.model.variables))
            return t[0], t[2], t[3], t[4], t[5], t[6], t[7]

    @tf.function
    def run_fisher(self, F, C, Cinv, μ, dμ_dθ, validate=True):
        """Calculation of the Fisher information by passing data through model

        Parameters
        __________
        F : TF tensor float (n_params, n_params)
            Fisher information matrix
        C : TF tensor float (n_summaries, n_summaries)
            covariance of summaries
        Cinv : TF tensor float (n_summaries, n_summaries)
            inverse covariance of summaries
        μ : TF tensor float (n_summaries)
            mean of the summaries
        dμ_dθ : TF tensor float (n_params, n_summaries)
            derivative of mean summaries with respect to the parameters
        validate : bool
            whether the data is validation data or not

        Returns
        _______
        F : TF tensor float (n_params, n_params)
            Fisher information matrix
        C : TF tensor float (n_summaries, n_summaries)
            covariance of summaries
        Cinv : TF tensor float (n_summaries, n_summaries)
            inverse covariance of summaries
        μ : TF tensor float (n_summaries)
            mean of the summaries
        dμ_dθ : TF tensor float (n_params, n_summaries)
            derivative of mean summaries with respect to the parameters

        Calls
        _____
        calculate_fisher(tensor, tensor)
            calculates the statistics necessary for Fisher from the summaries
        """
        if validate:
            data = self.validation_data
            derivative = self.validation_derivative
        else:
            data = self.data
            derivative = self.derivative
        x = self.model(data)
        dx_dθ = tf.reshape(
            self.model(
                tf.reshape(
                    derivative,
                    (self.n_d * 2 * self.n_params,) + self.input_shape)),
            (self.n_d, 2, self.n_params, self.n_summaries))
        return self.calculate_fisher(x, dx_dθ)

    @tf.function
    def fast_train(self, F, C, Cinv, μ, dμ_dθ, reg, r):
        """Automatic calculation of gradients for updating weights

        The Fisher information is maximised by automatically calculating the
        derivative of the logarithm of the determinant of the Fisher matrix
        regularised by the Frobenius norm of the elementwise difference of the
        summary covariance and the inverse covariance of the summaries from the
        identity matrix. The regulariser is necessary to set the scale of the
        summaries, i.e. the set of summaries which are preferably orthogonal
        with covariance of I. The strength of the regularisation is smoothly
        reduced as the summaries approach a covariance of the identity matrix
        so that preference is given to optimising the Fisher information
        matrix.

        The gradients are calculated end-to-end in the graph and so it
        necessitates relatively small inputs. If the inputs are too big for
        fast_train(), then the data needs to be initialised as a dataset,

        Parameters
        __________
        F : TF tensor float (n_params, n_params)
            Fisher information matrix
        C : TF tensor float (n_summaries, n_summaries)
            covariance of summaries
        Cinv : TF tensor float (n_summaries, n_summaries)
            inverse covariance of summaries
        μ : TF tensor float (n_summaries)
            mean of the summaries
        dμ_dθ : TF tensor float (n_params, n_summaries)
            derivative of mean summaries with respect to the parameters
        reg : TF tensor float ()
            value of regulariser
        r : TF tensor float ()
            value of the dynamical coupling strength of the regulariser

        Returns
        _______
        F : TF tensor float (n_params, n_params)
            Fisher information matrix
        C : TF tensor float (n_summaries, n_summaries)
            covariance of summaries
        Cinv : TF tensor float (n_summaries, n_summaries)
            inverse covariance of summaries
        μ : TF tensor float (n_summaries)
            mean of the summaries
        dμ_dθ : TF tensor float (n_params, n_summaries)
            derivative of mean summaries with respect to the parameters
        reg : TF tensor float ()
            value of regulariser
        r : TF tensor float ()
            value of the dynamical coupling strength of the regulariser

        Calls
        _____
        run_fisher(tensor, tensor, tensor, tensor, tensor, bool)
            calulates the Fisher information matrix from scratch
        calculate_loss(tensor, tensor, tensor)
            calculates the loss function and the regularisation
        """
        with tf.GradientTape() as tape:
            F, C, Cinv, μ, dμ_dθ = self.run_fisher(F, C, Cinv, μ, dμ_dθ,
                                                   validate=False)
            Λ, regularisation, r = self.calculate_loss(F, C, Cinv)
        gradients = tape.gradient(Λ, self.model.variables)
        self.optimiser.apply_gradients(zip(gradients, self.model.variables))
        return F, C, Cinv, μ, dμ_dθ, reg, r

    def get_regularisation_rate(self, λ, ϵ):
        """Calculate the dynamical coupling stregth turn over rate

        Paramaters
        __________
        λ : float
            coupling strength of the regulariser term when far from identity
        ϵ : float
            closeness parameter describing how far covariance is from identity
        """
        self.λ = tf.convert_to_tensor(
            self.u.type_checking(λ, 1., "λ"),
            dtype=self.dtype)
        ϵ = tf.convert_to_tensor(
            self.u.type_checking(ϵ, 1., "ϵ"),
            dtype=self.dtype)
        self.α = -tf.divide(
            tf.math.log(
                tf.add(
                    tf.multiply(
                        tf.subtract(
                            λ,
                            tf.convert_to_tensor(1., dtype=self.dtype)),
                        ϵ),
                    tf.divide(
                        tf.square(ϵ),
                        tf.add(
                            tf.convert_to_tensor(1., dtype=self.dtype),
                            ϵ)))),
            ϵ)

    def get_MLE(self, d):
        """Calculates MLE from score compression-like expansion

        Parameters
        __________
        d : TF tensor float (None,) + input_shape

        Returns
        _______
        MLE : TF tensor float (None, n_params)
            value of the MLE estimated via the IMNN
        """
        self.Finv = tf.linalg.inv(self.F)
        return tf.add(
            self.θ_fid,
            tf.einsum(
                "ij,kj->ki",
                self.Finv,
                tf.einsum(
                    "ij,kj->ki",
                    self.dμ_dθ,
                    tf.einsum(
                        "ij,kj->ki",
                        self.Cinv,
                        tf.subtract(
                            self.model(d),
                            self.μ)))))

    def fit(self, n_iterations, λ=None, ϵ=None, reset=False,
            patience=None, checkpoint=None, min_iterations=None,
            tqdm_notebook=True):
        """Fitting routine for IMNN

        Can reset model if training goes awry and clear diagnostics.
        Diagnostics are collected after one whole pass through the data.
        Validation can also be done if validation set is defined.

        Parameters
        __________
        n_iterations : int
            number of complete passes through the data
        λ : float
            coupling strength for the regularisation
        ϵ : float
            closness condition for covariance to the identity
        reset : bool
            whether to reset weights of the model and clear diagnostic values
        patience : int
            number of iterations to check for early stopping
        checkpoint : int
            number of iterations at which to checkpoint
        min_iterations : int
            number of initial iterations before using patience
        tqdm_notebook : bool
            whether to use a notebook style tqdm progress bar

        Calls
        _____
        initialise_history()
            sets up dictionary of lists for collecting training diagnostics

        IMNN.utils.utils.isnotebook()
            checks whether IMNN being trained in jupyter notebook

        """
        if reset:
            self.initialise_history()
            self.model.reset_states()
        if n_iterations is None:
            n_iterations = int(1e10)
        if (self.λ is None) or (self.α is None):
            self.get_regularisation_rate(λ, ϵ)
        elif (λ is not None) and (ϵ is not None):
            self.get_regularisation_rate(λ, ϵ)
        else:
            self.u.regularisation_error()

        if checkpoint is not None:
            if not self.save:
                if self.verbose:
                    self.u.save_error()
            to_checkpoint = True
            self.model.save_weights(self.filename + "/model_weights.h5")
        else:
            to_checkpoint = False
        if patience is not None:
            if not self.save:
                self.u.save_error()
            else:
                if self.verbose:
                    print("Using patience length of " + str(patience) +
                          ". Maximum number of training iterations is " +
                          str(n_iterations) + ".")
                    print("Saving current model in " + self.filename)
                self.model.save(self.filename)
                self.model.save_weights(self.filename + "/model_weights.h5")
                patience_counter = 0
                this_iteration = 0
                calculate_patience = True
                min_reached = False
                if min_iterations is None:
                    min_iterations = 1
                if self.validate:
                    patience_criterion = "val_det_F"
                else:
                    patience_criterion = "det_F"
                if checkpoint is None:
                    checkpoint = 1
                    to_checkpoint = True
        else:
            calculate_patience = False

        if self.u.isnotebook(tqdm_notebook):
            bar = tqdm.tnrange(n_iterations, desc="Iterations")
        else:
            bar = tqdm.trange(n_iterations, desc="Iterations")
        for iterations in bar:
            self.F, self.C, self.Cinv, self.μ, self.dμ_dθ, self.reg, self.r = \
                self.trainer(self.F, self.C, self.Cinv, self.μ, self.dμ_dθ,
                             self.reg, self.r)
            self.history["det_F"].append(tf.linalg.det(self.F).numpy())
            self.history["det_C"].append(tf.linalg.det(self.C).numpy())
            self.history["det_Cinv"].append(tf.linalg.det(self.Cinv).numpy())
            self.history["dμ_dθ"].append(self.dμ_dθ.numpy())
            self.history["reg"].append(self.reg.numpy())
            self.history["r"].append(self.r.numpy())
            postfix_dictionary = {
                "det_F": self.history["det_F"][-1],
                "det_C": self.history["det_C"][-1],
                "det_Cinv": self.history["det_Cinv"][-1],
                "r": self.history["r"][-1]}
            if self.validate:
                self.F, self.C, self.Cinv, self.μ, self.dμ_dθ = \
                    self.validater(self.F, self.C, self.Cinv, self.μ,
                                   self.dμ_dθ, validate=True)
                self.history["val_det_F"].append(tf.linalg.det(self.F).numpy())
                self.history["val_det_C"].append(tf.linalg.det(self.C).numpy())
                self.history["val_det_Cinv"].append(
                    tf.linalg.det(self.Cinv).numpy())
                self.history["val_dμ_dθ"].append(self.dμ_dθ.numpy())
                postfix_dictionary["val_det_F"] = self.history["val_det_F"][-1]
                postfix_dictionary["val_det_C"] = self.history["val_det_C"][-1]
                postfix_dictionary["val_det_Cinv"] = \
                    self.history["val_det_Cinv"][-1]
            if to_checkpoint:
                if calculate_patience:
                    if min_reached:
                        if (self.history[patience_criterion][-1]
                                <= self.history[patience_criterion][-2]):
                            if patience_counter > patience:
                                print("Reached " + str(patience)
                                      + " steps without increasing "
                                      + patience_criterion
                                      + ". Resetting weights to iteration "
                                      + str(this_iteration) + ".")
                                self.model.load_weights(
                                    self.filename + "/model_weights.h5")
                                break
                            else:
                                patience_counter += 1
                        else:
                            patience_counter = 0
                            if iterations % checkpoint == 0:
                                this_iteration = iterations
                                self.model.save_weights(
                                    self.filename + "/model_weights.h5")
                    else:
                        if iterations > min_iterations:
                            min_reached = True
                        if iterations % checkpoint == 0:
                            this_iteration = iterations
                            self.model.save_weights(
                                self.filename + "/model_weights.h5")
                    postfix_dictionary["patience"] = patience_counter
                else:
                    if iterations % checkpoint == 0:
                        this_iteration = iterations
                        self.model.save_weights(
                            self.filename + "/model_weights.h5")
            bar.set_postfix(postfix_dictionary)
