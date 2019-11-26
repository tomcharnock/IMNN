"""Information maximising neural network
This module provides the methods necessary to build and train an information
maximising neural network to optimally compress data down to the number of
model parameters.

TODO
____
Still some docstrings which need finishing
Sequential training for large data
Use precomputed external covariance and derivatives
"""


__version__ = '0.2a1'
__author__ = "Tom Charnock"


import tensorflow as tf
import tqdm
from utils.utils import utils


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
    n_params : int
        number of parameters in physical model
    n_summaries : int
        number of summaries to compress data to
    n_s : int
        number of simulations to calculate summary covariance
    n_p : int
        number of derivatives simulations to calculate derivative of mean
    single_dataset : bool
        whether multiple datasets are needed for fiducial and derivative sims
    numerical : bool
        whether numerical derivative is used for training data
    test_numerical : bool {None}
        whether numerical derivative is used for validation data
    use_external : bool
        whether external summaries are used for training data
    test_use_external : bool {None}
        whether external summaries are used for validation data
    dΔμ_dx : TF tensor float (n_s, n_s, n_summaries, n_summaries)
        derivative of the mean training summaries with respect to the summaries
    n_st : TF tensor float (1, )
        number of simulations to calculate summary covariance as tensor float
    n_sm1 : TF tensor float (1, )
        1 mines number of simulations to calculate unbiased summary covariance
    model : TF model - keras or other
        neural network to do the compression defined using TF or keras
    optimiser : TF optimiser - keras or other
        optimisation operation to do weight updates, defined using TF or keras
    θ_fid : TF tensor float (n_params,)
        fiducial parameter values for training dataset
    test_θ_fid : TF tensor float {None} (n_params,)
        fiducial parameter values for validation dataset
    δθ : TF tensor float {None} (n_params,)
        parameter differences for numerical derivatives
    d2μ_dθdx : TF tensor (n_d, n_params, n_summaries, n_summaries, n_params)
        derivative of mean summaries with respect to the numerical summaries
    test_δθ : TF tensor float {None} (n_params,)
        parameter differences for numerical derivatives of validation dataset
    dataset : TF dataset
        TF dataset for data input
    test_dataset : TF dataset {None}
        TF dataset for validation data input
    derivative_dataset : TF dataset {None}
        TF dataset for numerical derivative simulations
    test_derivative_dataset : TF dataset {None}
        TF dataset for numerical derivative validation simulations
    F : TF tensor float (1 ,)
        collector for determinant of Fisher matrix
    C : TF tensor float (1 ,)
        collector for determinant of covariance of summaries
    Cinv : TF tensor float (1 ,)
        collector for determinant of inverse covariance of summaries
    dμ_dθ : TF tensor float [n_params, n_summaries]
        collector for determinant of derivative of mean summaries wrt params
    reg : TF tensor float (1 ,)
        collector for value of regulariser
    r : TF tensor float (1 ,)
        collector for value of coupling strength of the regulariser
    MLE_F : TF tensor float {None} (n_params, n_params)
        Fisher information matrix for using in ABC
    MLE_Finv : TF tensor float {None} (n_params, n_params)
        inverse Fisher information matrix for calculating MLE
    MLE_Cinv : TF tensor float {None} (n_summaries, n_summaries)
        inverse covariance matrix of summaries for calculating MLE
    MLE_dμ_dθ : TF tensor float {None} (n_params, n_summaries)
        derivative of the mean summaries with respect to parameters for MLE
    MLE_μ : TF tensor float {None} (n_summaries,)
        mean summary value for calculating MLE
    MLE_θ_fid : None (n_params,)
        fiducial parameter values for calculating MLE
    history : dict
        history object for saving training statistics.
    """
    def __init__(self, n_params, n_summaries, n_covariance_sims,
                 n_derivative_sims, dtype=tf.float32, verbose=True):
        """Initialises attributes and calculates useful constants

        Parameters
        __________
        n_params : int
            number of parameters in physical model
        n_summaries : int
            number of summaries to compress data to
        n_covariance_sims : int
            number of simulations to calculate summary covariance
        n_covariance_sims : int
            number of derivatives simulations to calculate derivative of mean
        dtype : TF type
            32 bit or 64 TensorFlow tensor floats (default tf.float32)
        verbose : bool
            whether to use verbose outputs in error checking module

        Calls
        _____
        initialise_attributes(int, int, int, int, tf.dtype)
            Initialises all attributes and sets necessary constants
        """
        self.u = utils(verbose=verbose)
        self.initialise_attributes(n_params, n_summaries, n_covariance_sims,
                                   n_derivative_sims, dtype)

    def initialise_attributes(self, n_params, n_summaries, n_covariance_sims,
                              n_derivative_sims, dtype):
        """Initialises all attributes and sets necessary constants

        All attributes are set to None before they are loaded when
        necessary. The number of parameters and summaries are set and
        the number of simulations needed for the covariance and derivatives
        of the mean summaries is set.

        Parameters
        __________
        n_params : int
            number of parameters in physical model
        n_summaries : int
            number of summaries to compress data to
        n_covariance_sims : int
            number of simulations to calculate summary covariance
        n_covariance_sims : int
            number of derivatives simulations to calculate derivative of mean
        dtype : TF type
            32 bit or 64 TensorFlow tensor floats (default tf.float32)

        Calls
        _____
        IMNN.utils.utils.positive_integer(int, str) -> int
            checks whether parameter is positive integer and error otherwise
        IMNN.utils.utils.check_num_datasets(int, int) -> bool
            checks whether to use a single dataset for derivatives and data
        initialise_history()
            sets up dictionary of lists for collecting training diagnostics
        load_useful_constants()
            makes TF tensors for necessary objects which can be precomputed
        """
        if dtype == tf.float64:
            self.dtype = tf.float32
            self.itype = tf.int64
        else:
            self.dtype = tf.float32
            self.itype = tf.int32

        self.n_params = self.u.positive_integer(n_params, "n_params")
        self.n_summaries = self.u.positive_integer(n_summaries, "n_summaries")
        self.n_s = self.u.positive_integer(
            n_covariance_sims, "n_covariance_sims")
        self.n_d = self.u.positive_integer(
            n_derivative_sims, "n_derivative_sims")
        self.single_dataset = self.u.check_num_datasets(self.n_s, self.n_d)

        self.numerical = None
        self.test_numerical = None
        self.use_external = None
        self.test_use_external = None
    #    self.sims_at_once = None
    #    self.loop_sims = None

        self.dΔμ_dx = None
        self.n_st = None
        self.n_sm1 = None
        self.identity = None

        self.model = None
        self.optimiser = None

        self.θ_fid = None
        self.test_θ_fid = None
        self.δθ = None
        self.d2μ_dθdx = None
        self.test_δθ = None
        self.dataset = None
        self.test_dataset = None
        self.derivative_dataset = None
        self.test_derivative_dataset = None
    #    self.indices = None
    #    self.derivative_indices = None

        self.MLE_F = None
        self.MLE_Finv = None
        self.MLE_Cinv = None
        self.MLE_dμ_dθ = None
        self.MLE_μ = None
        self.MLE_θ_fid = None

        self.initialise_history()
        self.load_useful_constants()

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
            val_reg - value of the regularisation term from validation set
            r - value of the coupling strength of the regulariser
            val_r - value of validation coupling strength of the regulariser
        """
        self.history = {
            "det_F": [],
            "val_det_F": [],
            "det_C": [],
            "val_det_C": [],
            "det_Cinv": [],
            "val_det_Cinv": [],
            "dμ_dθ": [],
            "val_dμ_dθ": [],
            "reg": [],
            "val_reg": [],
            "r": [],
            "val_r": []
        }

    def load_useful_constants(self):
        """Makes TF tensors for necessary objects which can be precomputed

        Sets up the loop variable tensors for training and validation and
        calculates the derivative of the mean summaries with respect to outputs
        which can be precomputed. Also makes tensor float version of number of
        simulations for summary covariance and this value minus 1 for unbiased
        covariance calculation. The identity matrix is also defined.

        Calls
        _____
        get_dΔμ_dx() -> TF tensor float (n_s, n_s, n_summaries, n_summaries)
            derivative of the mean training summaries wrt the summaries
        get_n_s_minus_1() -> TF tensor float (1, )
            subtracts 1 from the value of the number of sims for covariance
        """
        self.F = tf.Variable(0., dtype=self.dtype)
        self.C = tf.Variable(0., dtype=self.dtype)
        self.Cinv = tf.Variable(0., dtype=self.dtype)
        self.dμ_dθ = tf.zeros((self.n_params, self.n_summaries),
                              dtype=self.dtype)
        self.reg = tf.Variable(0., dtype=self.dtype)
        self.r = tf.Variable(0., dtype=self.dtype)

        self.dΔμ_dx = self.get_dΔμ_dx()
        self.n_st = tf.Variable(self.n_s, dtype=self.dtype, name="n_s")
        self.n_sm1 = self.get_n_s_minus_1()
        self.identity = tf.eye(self.n_summaries)

    def get_dΔμ_dx(self):
        """Builds derivative of the mean training summaries wrt the summaries

        The difference between the derivative of the mean of summaries wrt the
        summaries and the derivative of the summaries wrt the summaries can be
        calculated by building the Kronecker delta which is
        $\\delta_{ab}\\delta_{ij} = \\frac{\\partial x_a^i}{\\partial x_b^j}$
        where a and b label the summary and i and j label the simulation.
        We then take the difference between this and its mean across the i
        simulations. This is needed for calculating the derivative of the
        covariance with respect to the summaries for the backpropagation.

        Returns
        _______
        TF tensor float (n_s, n_s, n_summaries, n_summaries)
            derivative of the mean training summaries wrt the summaries
        """
        dx_dx = tf.einsum(
            "ij,kl->ijkl",
            tf.eye(self.n_s, self.n_s),
            tf.eye(self.n_summaries, self.n_summaries),
            name="derivative_of_summaries_wrt_summaries")
        dμ_dx = tf.reduce_mean(dx_dx, axis=0, keepdims=True,
                               name="derivative_of_mean_x_wrt_x")
        return tf.subtract(dx_dx, dμ_dx,
                           name="derivative_of_diff_mean_x_wrt_x")

    def get_n_s_minus_1(self):
        """Subtracts 1 from the value of the number of sims for covariance

        Returns
        _______
        TF tensor float (1,)
            (n_s - 1) as a float tensor
        """
        return tf.subtract(
            tf.cast(
                self.n_s,
                self.dtype,
                name="number_of_simulations_float"),
            1.,
            name="number_of_simulations_minus_1")

    def get_d2μ_dθdx(self, δθ):
        """Calculate derivative of mean summaries wrt the numerical summaries

        The derivative of the mean summaries with respect to the summaries of
        the simulations for the numerical derivatives can be calculated
        knowning only the width of the parameter values using
        $\\frac{\\partial^2\\mu_\\mathscr{f}}{\\partial x\\partial\\theta} =
         \\frac{1}{2\\delta\\theta_\\alpha n_d}\\sum_i\\delta_{ab}
         \\delta_{\\alpha\\beta}\\delta_{ij}$
        where
        $\\delta_{ab}\\delta_{\\alpha\\beta}\\delta_{ij} =
            \\frac{\\partial x^{i\\alpha}_a}{\\partial x^{j\\beta}}_b}$
        in which a and b label the summary, $\\alpha$ and $\\beta$ label the
        which parameter the numerical derivative is with respect to and i and j
        label the simulation.

        Parameters
        __________
        δθ : TF tensor float (n_params,)
            parameter differences for numerical derivatives

        Returns
        _______
        TF tensor float (n_d, n_params, n_summaries, n_summaries, n_params)
            derivative of mean summaries wrt the numerical summaries
        """
        dxa_dxb = tf.einsum(
            "ij,kl,mn->ijklmn",
            tf.eye(self.n_d, self.n_d),
            tf.eye(self.n_params, self.n_params),
            tf.eye(self.n_summaries, self.n_summaries),
            name="derivative_of_x_wrt_x_for_derivatives")
        return tf.reduce_mean(
            tf.einsum(
                "ijklmn,l->ijkmnl",
                dxa_dxb,
                δθ,
                name="derivative_of_x_wrt_x_and_parameters"),
            axis=0,
            name="derivative_of_mean_x_wrt_x_and_parameters")

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
        check_model(int, int)
            prints warning that model should be correct size (might upgrade to
            real model checking)
        """
        self.u.check_model(self.n_params, self.n_summaries)
        self.model = model
        self.optimiser = optimiser

    def load_fiducial(self, θ_fid, train):
        """Loads the fiducial parameters into a TF tensor

        Checks that the fiducial parameter values are of the correct shape
        and makes a tensor from it. This is then loaded into the training or
        validation attribute.

        Parameters
        __________
        θ_fid : np.ndarray (n_params)
            fiducial parameter values
        train : bool
            whether fiducial parameter values are for validation or training

        Calls
        _____
        IMNN.utils.utils.fiducial_check(np.ndarray, int)
            checks the size of the fiducial parameters array is correct
        """
        self.u.fiducial_check(θ_fid, self.n_params)
        if train:
            self.θ_fid = tf.Variable(θ_fid, dtype=self.dtype, trainable=False,
                                     name="fiducial")
        else:
            self.test_θ_fid = tf.Variable(θ_fid, dtype=self.dtype,
                                          trainable=False,
                                          name="test_fiducial")

    def check_derivative(self, dd_dθ, δθ, train):
        """Checks whether numerical derivative is used

        Checks whether the numerical parameter difference is defined, and if it
        is then sets the attribute to use numerical derivatives. If doing
        numerical derivatives then the parameter widths are loaded as tensors
        and the derivative of the mean of the numerical derivative summaries
        with respect to the summaries is calculated.

        TODO
        ____
        Should check that the shape of the derivatives is correct if δθ is
        passed. If δθ is passed then numpy array of derivative simulations
        should be be (n_d, 2, n_params) + input_shape where the first element
        of the second axis is the lower derivatives and the second element is
        the upper derivatives.

        Parameters
        __________
        dd_dθ : np.ndarray (n_s, n_params, ...) or (n_d, 2, n_params, ...)
            array of the derivatives of the simulation (could be numerical)
        δθ : np.ndarray (n_params,) {None}
            parameter differences for numerical derivatives if using
        train : bool
            whether fiducial parameter values are for validation or training

        Returns
        _______
        bool
            whether numerical derivatives should be done

        Calls
        _____
        IMNN.utils.utils.bool_none(any) -> bool
            checks whether input exists
        IMNN.utils.utils.delta_check(np.ndarray, int)
            checks wether parameter differences has correct shape (n_params,)
        get_d2μ_dθdx(TF tensor float) -> TF tensor float
                (n_d, n_params, n_summaries, n_summaries, n_params)
            derivative of mean summaries wrt the numerical summaries
        """
        numerical = self.u.bool_none(δθ)
        if numerical:
            self.u.delta_check(δθ, self.n_params)
            δθ_tensor = tf.Variable(
                1. / (2. * δθ), dtype=self.dtype, trainable=False,
                name="delta_theta")
        if train:
            self.numerical = numerical
            if numerical:
                self.δθ = δθ_tensor
                self.d2μ_dθdx = self.get_d2μ_dθdx(self.δθ)
        else:
            self.test_δθ = δθ_tensor
            self.test_numerical = numerical
        return numerical

    # def to_loop_sims(self, sims_at_once, train):
    #    loop_sims = bool_none(sims_at_once)
    #    if train:
    #        self.loop_sims = loop_sims
    #    else:
    #        self.test_loop_sims = loop_sims
    #    return loop_sims

    def use_external_summaries(self, external_summaries, train):
        """Checks whether external summaries are used

        It is possible to use external summaries as informative summaries
        alongside the IMNN summaries. This function checks whether these
        summaries have been passed.

        Parameters
        __________
        external_summaries : np.ndarray (n_s, n_external_summaries)
            array of other informative summaries to be included
        train : bool
            whether fiducial parameter values are for validation or training

        Returns
        _______
        bool
            whether external summaries should be used

        Calls
        _____
        IMNN.utils.utils.bool_none(any) -> bool
            checks whether input exists
        """
        use_external = self.u.bool_none(external_summaries)
        if train:
            self.use_external = use_external
        else:
            self.test_use_external = use_external
        return use_external

    def build_dataset(self, data, batchsize=None, shufflesize=None):
        """Create tensorflow dataset and split into batches and shuffle

        Parameters
        __________
        data : np.ndarray or (np.ndarray, ...)
            the data to be placed into the tensorflow dataset
        batchsize : int
            the size of the batch for calculating covariance or mean derivative
        shufflesize : int
            how many simulations should be shuffled (should be size of data[0])

        Returns
        _______
        dataset : TF dataset
            the tensorflow dataset containing the data
        """
        dataset = tf.data.Dataset.from_tensor_slices(data)
        if batchsize is not None:
            dataset = dataset.batch(batchsize)
        elif shufflesize is not None:
            dataset = dataset.shuffle(shufflesize)
        return dataset

    def setup_dataset(self, θ_fid, d, dd_dθ, δθ=None, external_summaries=None,
                      external_derivatives=None, # sims_at_once=None,
                      train=True):
        """Builds TF datasets for training or validation sets

        By passing data to the function all options for training and validation
        are set, including batch sizing, whether to do numerical derivatives
        whether external summaries should be included. Once constructed the
        datasets are set as module attributes.

        TODO
        ____
        For very large simulations we would need to compute the summaries a
        few at a time and collect the derivatives and summaries to compute the
        Fisher information and backpropagation sequentially. This would take
        a parameter sims_at_once to say how many simulations could be processed
        at once and would set the loop_sims attribute. This is mostly
        implemented but is commented out.
        It would be possible to have external covariances and derivatives
        precomputed and therefore not need to be computed on every iteration.
        This would save computation, especially when the dimension of the
        external summaries is large. In this case we would only need to
        calculate the covariance of the set of external summaries with the IMNN
        summaries Cov[{s}, {x}] and never Cov[{s}, {s}]. The values of
        Cov[{s}, {x}] could just be appended to the outer block of the
        covariance matrix which would be cheaper. The numerical derivative of
        the external summaries could also be precomputed and just appended.

        Parameters
        __________
        θ_fid : np.ndarray (n_params,)
            parameter differences for numerical derivatives
        d : np.ndarray (n_s,) + input_shape
            simulations at fiducial parameter values for calculating covariance
        dd_dθ : np.ndarray (n_s, n_params, ...) or (n_d, 2, n_params, ...)
            array of the derivatives of the simulation (could be numerical)
        δθ : np.ndarray (n_params,) {None}
            parameter differences for numerical derivatives if using
        external_summaries : np.ndarray (n_s, n_external_summaries) {None}
            set of informative summaries for each simulation
        external_derivatives : np.ndarray (n_s, n_params, n_external_summaries)
                               or (n_s, 2, n_params, n_external_summaries)
            derivative of the informative summaries wrt model parameters
        train : bool
            whether fiducial parameter values are for validation or training

        Calls
        _____
        load_fiducial(np.ndarray, bool)
            loads the fiducial parameters into a TF tensor
        use_external_summaries(np.ndarray, np.ndarray, bool) -> bool
            checks whether external summaries are used
        IMNN.utils.utils.batch_warning(int, int, bool) -> int
            checks whether the batchsize is valid given input data
        IMNN.utils.utils.size_check(int, int, str, str)
            checks wether two datasets have compatible sizes
        IMNN.util.utils.numericial_size_check(int, int, bool)
            checks whether the simulations for numercial derivatives is correct
        build_dataset(np.ndarray, {int}, {int}) -> TF dataset
            Create tensorflow dataset and split into batches and shuffle
        """

        self.load_fiducial(θ_fid, train)
        numerical = self.check_derivative(dd_dθ, δθ, train)
    #    loop_sims = self.to_loop_sims(sims_at_once, train)
        use_external = self.use_external_summaries(external_summaries, train)
        n_batch = self.u.batch_warning(d.shape[0], self.n_s, train)

        if (not self.single_dataset) and (not use_external):
            data = d
        else:
            data = (d,)

        if self.single_dataset:
            self.u.size_check(dd_dθ.shape[0], d.shape[0], "dd_dθ", "d")
            n_d_batch = n_batch
            data += (dd_dθ,)
        else:
            if numerical:
                n_d_batch = self.u.batch_warning(dd_dθ.shape[0], self.n_d,
                                                 train, derivative=True)
            else:
                self.u.numerical_size_check(dd_dθ.shape[0], d.shape[0],
                                            numerical)
                n_d_batch = n_batch
            d_data = dd_dθ

        if use_external:
            self.u.size_check(external_summaries.shape[0], d.shape[0],
                              "external_summaries", "d")
            data += (external_summaries,)
            if self.single_dataset:
                self.u.size_check(external_derivatives.shape[0], d.shape[0],
                                  "external_derivatives", "d")
                data += (external_derivatives,)
            else:
                if numerical:
                    self.u.size_check(external_derivatives.shape[0],
                                      dd_dθ.shape[0], "external_derivatives",
                                      "dd_dθ")
                    d_data = (d_data, external_derivatives)

        dataset = self.build_dataset(data, batchsize=self.n_s,
                                     shufflesize=self.n_s * n_batch)
        if not self.single_dataset:
            d_dataset = self.build_dataset(d_data, batchsize=self.n_d,
                                           shufflesize=self.n_d * n_d_batch)
    #    if loop_sims:
    #       def loop_batch(*x):
    #           return self.loop_batch(sims_at_once, *x)
    #       ind = tf.expand_dims(tf.range(self.n_s, dtype=self.itype), 1)
    #       indices = self.build_dataset(ind, batchsize=sims_at_once)
    #       if self.single_dataset:
    #           dataset = dataset.map(loop_batch)
    #       else:
    #           d_ind = tf.expand_dims(tf.range(self.n_d,
    #                                           dtype=self.itype), 1)
    #           d_indices = self.build_dataset(d_ind, batchsize=sims_at_once)
    #           if use_external:
    #               dataset = dataset.map(loop_batch)
    #               d_dataset = dataset.map(loop_batch)
    #           else:
    #               dataset = dataset.map(
    #                   lambda x: (tf.data.Dataset.from_tensor_slices(x)
    #                              .batch(sims_at_once).repeat(2)))
    #               d_dataset = d_dataset.map(
    #                   lambda x: (tf.data.Dataset.from_tensor_slices(x)
    #                              .batch(sims_at_once).repeat(2)))

        if train:
            self.dataset = dataset
    #        if loop_sims:
    #           self.indices = indices
            if not self.single_dataset:
                self.derivative_dataset = d_dataset
    #            if loop_sims:
    #               self.derivative_indices = d_indices
        else:
            self.test_dataset = dataset
    #       if loop_sims:
    #           self.test_indices = indices
            if not self.single_dataset:
                self.test_derivative_dataset = d_dataset
    #            if loop_sims:
    #               self.test_derivative_indices = d_indices

    # def loop_batch(self, sims_at_once, *x):
    #     new_batch = tuple()
    #     for i in range(len(x)):
    #         new_batch += (tf.data.Dataset.from_tensor_slices(x[i])
    #                       .batch(self.sims_at_once)
    #                       .repeat(2),)
    #     return new_batch

    def fit(self, n_iterations, reset=False, validate=False):
        """Fitting routine for IMNN

        Can reset model if training goes awry and clear diagnostics.
        Diagnostics are collected after one whole pass through the data.
        Validation can also be done if validation set is defined.

        Parameters
        __________
        n_iterations : int
            number of complete passes through the data
        reset : bool
            whether to reset weights of the model and clear diagnostic values
        validate : bool
            whether to validate the model using preloaded dataset (not checked)

        Calls
        _____
        initialise_history()
            sets up dictionary of lists for collecting training diagnostics
        IMNN.utils.utils.isnotebook()
            checks whether IMNN being trained in jupyter notebook
        loop_train(tensor, tensor, tensor, tensor, tensor, tensor) ->
                tensor, tensor, tensor, tensor, tensor, tensor
            loop routine through entire training dataset to update weights
        loop_validate(tensor, tensor, tensor, tensor, tensor, tensor) ->
                tensor, tensor, tensor, tensor, tensor, tensor
            roop routine through entire validation dataset for diagnostics
        """
        if reset:
            self.initialise_history()
            self.model.reset_states()
        if self.u.isnotebook():
            bar = tqdm.tnrange(n_iterations, desc="Iterations")
        else:
            bar = tqdm.trange(n_iterations, desc="Iterations")
        for iterations in bar:
            self.F, self.C, self.Cinv, self.dμ_dθ, self.reg, self.r = \
                self.loop_train(self.F, self.C, self.Cinv, self.dμ_dθ,
                                self.reg, self.r)
            self.history["det_F"].append(self.F.numpy())
            self.history["det_C"].append(self.C.numpy())
            self.history["det_Cinv"].append(self.Cinv.numpy())
            self.history["dμ_dθ"].append(self.dμ_dθ.numpy())
            self.history["reg"].append(self.reg.numpy())
            self.history["r"].append(self.r.numpy())
            if validate:
                self.F, self.C, self.Cinv, self.dμ_dθ, self.reg, self.r = \
                    self.loop_validate(self.F, self.C, self.Cinv, self.dμ_dθ,
                                       self.reg, self.r)
                self.history["val_det_F"].append(self.F.numpy())
                self.history["val_det_C"].append(self.C.numpy())
                self.history["val_det_Cinv"].append(self.Cinv.numpy())
                self.history["val_dμ_dθ"].append(self.dμ_dθ.numpy())
                self.history["val_reg"].append(self.reg.numpy())
                self.history["val_r"].append(self.r.numpy())
                bar.set_postfix(
                    det_F=self.history["det_F"][-1],
                    det_C=self.history["det_C"][-1],
                    det_Cinv=self.history["det_Cinv"][-1],
                    r=self.history["r"][-1],
                    val_det_F=self.history["val_det_F"][-1],
                    val_det_C=self.history["val_det_C"][-1],
                    val_det_Cinv=self.history["val_det_Cinv"][-1],
                    val_r=self.history["val_r"][-1])
            else:
                bar.set_postfix(
                    det_F=self.history["det_F"][-1],
                    det_C=self.history["det_C"][-1],
                    det_Cinv=self.history["det_Cinv"][-1],
                    r=self.history["r"][-1])

    def automatic_train(self, x, dx_dθ, dx_dw, d2x_dwdθ, s=None, ds_dθ=None):
        """Automatic calculation of gradients for updating weights

        The Fisher information is maximised by automatically calculating the
        derivative of the logarithm of the determinant of the Fisher matrix
        regularised by the Frobenius norm of the elementwise difference of the
        summary covariance and the inverse covariance of the summaries from the
        identity matrix. The regulariser is necessary to set the scale of the
        summaries, i.e. the set of summaries which are preferably orthogonal
        with covariance of I. The strength of the regularisation is smoothly
        reduced as the summaries approach a covariance of I so that preference
        is given to optimising the Fisher information matrix.

        We calculate the analytic gradients of the loss function and the
        regularisation with respect to the outputs of the network and then
        update the weights using the optimisation scheme provided where the
        error is calculated using the chain rule
        $\\frac{\\partial\\Lambda}{\\partial w_j^l} =
            \\frac{\\partial\\Lambda}{\\partial x_i}
            \\frac{\\partial x_i}{\\partial w_j^l}$
        If numerical derivatives are used then we also need to include the
        response of the summaries for the derivatives on the network.

        Parameters
        __________

        Returns
        _______

        Calls
        _____
        """
        with tf.GradientTape() as tape:
            if self.numerical:
                tape.watch([x, dx_dθ])
            else:
                tape.watch(x)
            F, C, Cinv, dμ_dθ, _, _, _, _ = self.get_stats(
                x, dx_dθ, self.numerical, self.use_external, s=s,
                ds_dθ=ds_dθ, δθ=self.δθ)
            reg, _ = self.get_regularisation(C, Cinv)
            r, _ = self.get_r(reg)
            Λ = tf.subtract(tf.multiply(r, reg), tf.linalg.slogdet(F))
        if self.numerical:
            dΛ_dx, d2Λ_dxdθ = tape.gradient(Λ, [x, dx_dθ])
        else:
            dΛ_dx = tape.gradient(Λ, x)
        gradients = []
        for layer in range(len(self.model.variables)):
            gradients.append(
                tf.divide(
                    tf.einsum(
                        "ij,ij...->...",
                        dΛ_dx,
                        dx_dw[layer]),
                    tf.dtypes.cast(
                        self.n_s,
                        self.dtype)))
            if self.numerical:
                gradients[layer] = tf.add(
                    gradients[layer],
                    tf.divide(
                        tf.einsum(
                            "ijkl,ijkl...->...",
                            d2Λ_dxdθ,
                            d2x_dwdθ[layer]),
                        tf.dtypes.cast(
                            self.n_d,
                            self.dtype)))
        self.optimiser.apply_gradients(zip(gradients, self.model.variables))
        return F, C, Cinv, dμ_dθ, reg, r

    def unpack_data(self, data, use_external):
        """ Unpacks zipped data and returns in regular format

        For generality all data is drawn from zipped datasets. For readability
        the data is then unpacked here to be returned in the same format no
        matter what the zipped datasets are.

        Parameters
        __________
        data : tuple of TF tensor floats
            the zipped data to be unpacked
        use_external : bool
            whether external summaries are used

        Returns
        _______
        TF tensor float (n_s,) + input_shape
            Unpacked fiducial data simulations
        TF tensor float (n_s, n_params) + input_shape or
                        (n_d, 2, n_params) + input_shape
            Unpacked data for derivative or numerical derivative
        TF tensor float (n_s, n_external) {None}
            Unpacked external fiducial summaries if used
        TF tensor float (n_s, n_params, n_external) {None} or
                        (n_d, 2, n_params, n_external)
            Unpacked external summaries for derivative or numerical derivative
        """
        if self.single_dataset:
            d = data[0]
            dd_dθ = data[1]
            if use_external:
                s = data[2]
                ds_dθ = data[3]
            else:
                s = None
                ds_dθ = None
        else:
            if use_external:
                d = data[0]
                s = data[1]
                dd_dθ = data[2]
                ds_dθ = data[3]
            else:
                d = data[0]
                dd_dθ = data[1]
                s = None
                ds_dθ = None
        return d, dd_dθ, s, ds_dθ

    @tf.function
    def loop_train(self, F, C, Cinv, dμ_dθ, reg, r):
        """ Tensorflow dataset loop for training IMNN

        All data in data set in looped over, summarised (and the jacobian wrt
        the network parameters calculated) to calculate the Fisher information
        and the Jacobian of the regularised ln(det(F)) wrt to the network to
        optimise the weights to maximise the Fisher information.

        Note that we make use of a vectorised mapping on the whole batch to
        calculate the summaries and jacobian since we know that each input is
        independent. If we did not use this then the Jacobian calculation would
        massively dominate all the computation time.

        Parameters
        __________
        F : TF tensor float (1 ,)
            collector for determinant of Fisher matrix
        C : TF tensor float (1 ,)
            collector for determinant of covariance of summaries
        Cinv : TF tensor float (1 ,)
            collector for determinant of inverse covariance of summaries
        dμ_dθ : TF tensor float [n_params, n_summaries]
            collector for det of derivative of mean summaries wrt params
        reg : TF tensor float (1 ,)
            collector for value of regulariser
        r : TF tensor float (1 ,)
            collector for value of coupling strength of the regulariser

        Returns
        _______
        TF tensor float (1 ,)
            determinant of Fisher matrix
        TF tensor float (1 ,)
            determinant of covariance of summaries
        TF tensor float (1 ,)
            determinant of inverse covariance of summaries
        TF tensor float [n_params, n_summaries]
            determinant of derivative of mean summaries wrt params
        TF tensor float (1 ,)
            value of regulariser
        TF tensor float (1 ,)
            value of coupling strength of the regulariser

        Calls
        _____
        unpack_data(tuple of TF tensor float, bool)
                -> tensor, tensor, tensor {None}, tensor {None}
            unpacks zipped data and returns in regular format
        get_jacobian(TF tensor float, {bool})
            -> TF tensor float, {TF tensor float}, list of TF tensor
            summarises and calculates jacobian of outputs wrt network
        get_numerical_derivative_mean(TF tensor, TF tensor, bool, {TF tensor})
                -> TF tensor float
            calculates the numerical mean derivative of summaries wrt params
        get_summary_derivative_mean(TF tensor float, TF tensor float)
                -> TF tensor float
            append external summary derivatives to the IMNN summary derivatives
        get_covariance(TF tensor float) -> TF tensor, TF tensor, TF tensor
            calculates covariance, mean and difference of mean from zero for x
        get_score(TF tensor float, TF tensor float) -> TF tensor float
            calculates the product of the inverse covariance and dμ_dθ
        get_fisher_matrix(TF tensor float, TF tensor float) -> TF tensor float
            calculates the symmetric Fisher information matrix
        train(tensor, tensor, tensor, tensor, tensor,
                list, tensor, list, tensor, tensor) -> tensor, tensor
            caculates and applies the maximisation of the Fisher information
        """
        if self.single_dataset:
            loop = self.dataset
        else:
            loop = zip(self.dataset, self.derivative_dataset)
        for data in loop:
            d, dd_dθ, s, ds_dθ = self.unpack_data(data, self.use_external)
            x, dx_dw = tf.vectorized_map(
                lambda i: self.get_jacobian(
                    tf.expand_dims(i, 0)),
                d)
            if self.numerical:
                dx_dθ, d2x_dwdθ = tf.vectorized_map(
                    lambda i: self.get_jacobian(
                        tf.expand_dims(i, 0)),
                    dd_dθ)
            else:
                x, dx_dd, dx_dw = tf.vectorized_map(
                    lambda i: self.get_jacobian(
                        tf.expand_dims(i, 0),
                        derivative=True),
                    d)
                dx_dθ = None
                d2x_dwdθ = None
            F, C, Cinv, dμ_dθ, reg, r = self.automatic_train(
                x, dx_dθ, dx_dw, d2x_dwdθ, s=s, ds_dθ=ds_dθ)
        return tf.linalg.det(F), tf.linalg.det(C), tf.linalg.det(Cinv), \
            dμ_dθ, reg, r

    @tf.function
    def loop_validate(self, F, C, Cinv, dμ_dθ, reg, r):
        """ Tensorflow dataset loop for validating IMNN

        All data in validation data set in looped over and summarised to
        calculate the Fisher information and regularisation terms.

        Parameters
        __________
        F : TF tensor float (1 ,)
            collector for determinant of Fisher matrix
        C : TF tensor float (1 ,)
            collector for determinant of covariance of summaries
        Cinv : TF tensor float (1 ,)
            collector for determinant of inverse covariance of summaries
        dμ_dθ : TF tensor float [n_params, n_summaries]
            collector for det of derivative of mean summaries wrt params
        reg : TF tensor float (1 ,)
            collector for value of regulariser
        r : TF tensor float (1 ,)
            collector for value of coupling strength of the regulariser

        Returns
        _______
        TF tensor float (1 ,)
            determinant of Fisher matrix
        TF tensor float (1 ,)
            determinant of covariance of summaries
        TF tensor float (1 ,)
            determinant of inverse covariance of summaries
        TF tensor float [n_params, n_summaries]
            determinant of derivative of mean summaries wrt params
        TF tensor float (1 ,)
            value of regulariser
        TF tensor float (1 ,)
            value of coupling strength of the regulariser

        Calls
        _____
        unpack_data(tuple of TF tensor float, bool)
                -> tensor, tensor, tensor {None}, tensor {None}
            unpacks zipped data and returns in regular format
        get_fisher(tensor, tensor, bool, bool, {tensor}, {tensor}, {tensor})
                -> tensor, tensor, tensor, tensor, tensor, tensor
            generic calculation of all necessary components for Fisher matrix
        get_regularisation(tensor, tensor) -> tensor, tensor
            calculates the frobenius norm of |C-I|+|C^{-1}-I|
        get_r(tensor) -> tensor, tensor
            calculates the strength of the coupling of the regulariser
        """
        if self.single_dataset:
            loop = self.test_dataset
        else:
            loop = zip(self.test_dataset, self.test_derivative_dataset)
        for data in loop:
            d, dd_dθ, s, ds_dθ = self.unpack_data(data, self.test_use_external)
            F, C, Cinv, dμ_dθ, _, _ = self.get_fisher(
                d, dd_dθ, self.test_numerical, self.test_use_external, s=s,
                ds_dθ=ds_dθ, δθ=self.test_δθ)
            reg, _ = self.get_regularisation(C, Cinv)
            r, _ = self.get_r(reg)
        return tf.linalg.det(F), tf.linalg.det(C), tf.linalg.det(Cinv), \
            dμ_dθ, reg, r

    def get_fisher(self, d, dd_dθ, numerical, use_external, s=None, ds_dθ=None,
                   δθ=None):
        """Generic calculation of all necessary components for Fisher matrix

        Without calculating any gradients, the Fisher information can be
        calculated by passing the data and derivatives of the data through the
        neural network and then computing the covariance of the summaries and
        the numerical derivative of the summaries wrt the model parameters or
        using the chain rule with the Jacobian of the summaries wrt the data.

        Parameters
        __________
        d : TF tensor float (n_s,) + input_shape
            fiducial simulations from the dataset
        dd_dθ : TF tensor float (n_s, n_params) + input_shape or
                                (n_d, 2, n_params) + input_shape
            derivative of sims wrt params or sims for numerical derivative
        numerical : bool
            whether numerical derivative is used
        use_external : bool
            whether external summaries are used
        s : TF tensor float (n_s, n_external) {None}
            informative summaries at fiducial value
        ds_dθ : TF tensor float (n_s, n_params, n_external) {None} on
                                (n_s, 2, n_params, n_external) {None}
            informative summaries for derivative (numerical) wrt parameters
        δθ : TF tensor float {None} (n_params,)
            parameter differences for numerical derivatives

        Returns
        _______
        TF tensor float (n_params, n_params)
            Fisher information matrix
        TF tensor float (n_summaries, n_summaries) or
                        (n_summaries + n_external, n_summaries + n_external)
            covariance of the summaries
        TF tensor float (n_summaries, n_summaries) or
                        (n_summaries + n_external, n_summaries + n_external)
            inverse covariance of the summaries
        TF tensor float (n_params, n_summaries) or
                        (n_params, n_summaries + n_external)
            derivative of the mean of the summaries with respect to the params
        TF tensor float (1, n_summaries,) or (1, n_summaries + n_external,)
            mean of the summaries
        TF tensor float (n_params, n_summaries) or
                (n_params, n_summaries + n_external)
            product of inverse covariance and dμ_dθ

        Calls
        _____
        get_numerical_derivative_mean(TF tensor, TF tensor, bool, {TF tensor})
                -> TF tensor float
            calculates the numerical mean derivative of summaries wrt params
        get_summary_derivative_mean(TF tensor float, TF tensor float)
                -> TF tensor float
            append external summary derivatives to the IMNN summary derivatives
        get_covariance(TF tensor float) -> TF tensor, TF tensor, TF tensor
            calculates covariance, mean and difference of mean from summaries
        get_score(TF tensor float, TF tensor float) -> TF tensor float
            calculates the product of the inverse covariance and dμ_dθ
        get_fisher_matrix(TF tensor float, TF tensor float) -> TF tensor float
            calculates the symmetric Fisher information matrix
        """
        if numerical:
            x = self.model(d)
            dx_dθ = self.model(dd_dθ)
        else:
            with tf.TapeGradient() as tape:
                tape.watch(d)
                x = self.model(d)
            dx_dd = tape.batch_jacobian(x, d)
        F, C, Cinv, dμ_dθ, μ, score, _, _ = self.get_stats(
            x, dx_dθ, numerical, use_external, s=s, ds_dθ=ds_dθ, δθ=δθ)
        return F, C, Cinv, dμ_dθ, μ, score

    def get_stats(self, x, dx_dθ, numerical, external, s=None, ds_dθ=None,
                  δθ=None):
        """
        """
        if numerical:
            dμ_dθ = self.get_numerical_derivative_mean(dx_dθ, self.δθ,
                                                       external,
                                                       ds_dθ=ds_dθ)
        else:
            dμ_dθ = tf.divide(
                tf.einsum("ij...,ik...->kj", dx_dd, dd_dθ),
                self.n_st)
            if external:
                dμ_dθ = self.get_summary_derivative_mean(dμ_dθ, ds_dθ)
        C, μ, Δμ = self.get_covariance(x)
        Cinv = tf.linalg.inv(C)
        if external:
            C_n = C
            Cinv_n = Cinv
            C, _, _ = self.get_covariance(tf.concat([s, x], axis=1))
            Cinv = tf.linalg.inv(C)
        else:
            C_n = None
            Cinv_n = None
        score = self.get_score(Cinv, dμ_dθ)
        F = self.get_fisher_matrix(Cinv, dμ_dθ, score)
        return F, C, Cinv, dμ_dθ, μ, score, C_n, Cinv_n

    def get_jacobian(self, d, derivative=False):
        """ Calculates the summaries and Jacobian wrt to network parameters

        As part of a vectorised batch calculation the jacobian is calculated
        for the summaries wrt the network parameters per summary. This is much
        cheaper than calculating the whole Jacobian which is not needed. We can
        also calculate the Jacobian of the summaries with respect to the input
        if we are using the chain rule to calculate the derivative of the mean
        summaries with respect to the parameters.

        Parameters
        __________
        d : TF tensor float (1,) + input_shape
            fiducial sims or sims for numerical derivative from the dataset
        derivative : bool
            whether to calculate the Jacobian of the summaries wrt the data

        Returns
        _______
        TF tensor float (n_summaries,)
            a single element summary of the input data slice
        TF tensor float (n_summaries,) + input_shape {None}
            Jacobian of a summary wrt an input data slice
        list of TF tensor float
            Jacobian of a summary wrt the network parameters
        """
        with tf.GradientTape(persistent=True) as tape:
            if derivative:
                tape.watch(d)
            x = self.model(d)
        dx_dw = tape.jacobian(x, self.model.variables)
        if derivative:
            dx_dd = tape.jacobian(x, d)
            return tf.squeeze(x, 0), \
                tf.squeeze(dx_dd, (0, 2)), \
                [tf.squeeze(i, 0) for i in dx_dw]
        else:
            return tf.squeeze(x, 0), [tf.squeeze(i, 0) for i in dx_dw]

    def get_covariance(self, x):
        """Calculates covariance, mean and difference of mean from summaries

        Calculates the mean of the summaries and then finds the difference
        between the mean and each summary. This can then be used to calculate
        the covariance of the summaries.

        Parameters
        __________
        x : TF tensor float (n_s, n_summaries) or
                            (n_s, n_summaries + n_external)
            summaries of the input data (network and informative summaries)

        Returns
        _______
        TF tensor float (n_summaries, n_summaries) or
                        (n_summaries + n_external, n_summaries + n_external)
            covariance of the summaries
        TF tensor float (1, n_summaries,) or (1, n_summaries + n_external,)
            mean of the summaries
        TF tensor float (n_s, n_summaries,) or (n_s, n_summaries + n_external,)
            difference between mean of the summaries and the summaries
        """
        μ = tf.reduce_mean(
            x,
            axis=0,
            keepdims=True,
            name="mean")
        Δμ = tf.subtract(
            x,
            μ,
            name="centred_mean")
        C = tf.divide(
            tf.einsum(
                "ij,ik->jk",
                Δμ,
                Δμ,
                name="unnormalised_covariance"),
            self.n_sm1,
            name="covariance")
        return C, μ, Δμ

    def get_numerical_derivative_mean(self, dx_dθ, δθ, use_external,
                                      ds_dθ=None):
        """ Calculate the mean of the derivative of the summaries

        Numerically, we take away the simulations generated above the fiducial
        parameter values from the simulations generated below the fiducial
        parameter values and divide them by 2 times the difference between the
        upper and lower parameter values. If external informative summaries are
        provided then these are concatenated to the network summaries.

        Parameters
        _________
        dx_dθ : TF tensor float (n_d, 2, n_params, n_summaries)
            upper and lower parameter value summaries for numerical derivative
        δθ : TF tensor float (n_params,)
            parameter differences for numerical derivatives
        use_external : bool
            whether external summaries are used
        ds_dθ : TF tensor float (n_d, 2, n_params, n_external)
            upper and lower parameter value external summaries for derivative

        Returns
        _______
        TF tensor float (n_params, n_summaries) or
                        (n_params, n_summaries + n_external)
            derivative of the mean of the summaries with respect to the params
        """
        if use_external:
            lower = tf.concat([ds_dθ[:, 0, :, :], dx_dθ[:, 0, :, :]], axis=2)
            upper = tf.concat([ds_dθ[:, 1, :, :], dx_dθ[:, 1, :, :]], axis=2)
        else:
            lower = dx_dθ[:, 0, :, :]
            upper = dx_dθ[:, 1, :, :]
        return tf.reduce_mean(
            tf.multiply(
                tf.subtract(
                    upper,
                    lower),
                tf.expand_dims(tf.expand_dims(δθ, 0), 2)),
            axis=0,
            name="numerical_derivative_mean_wrt_parameters")

    def get_summary_derivative_mean(self, dμ_dθ_network, ds_dθ):
        """ Concatenate external derivative mean to network summary mean

        When using external summaries we concatenate the mean of the derivative
        of external summaries with respect to the parameters to the precomputed
        mean of the network summaries with respect to the parameters.

        Parameters
        __________
        dμ_dθ_network : TF tensor float (n_params, n_summaries)
            derivative of the mean network summaries wrt the parameters
        ds_dθ : TF tensor float (n_s, n_params, n_external)
            derivative of external summaries wrt the parameters

        Returns
        _______
        TF tensor float (n_params, n_summaries + n_external)
            derivative of all summaries wrt the parameters
        """
        dμ_dθ_external = tf.reduce_mean(ds_dθ, 0)
        return tf.concat([dμ_dθ_external, dμ_dθ_network], axis=2)

    def get_score(self, Cinv, dμ_dθ):
        """Product of inverse covariance and derivative mean wrt parameters

        This tensor is used multiple times so we calculate it and store it to
        be reused

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
        TF tensor float (n_params, n_summaries) or
                        (n_params, n_summaries + n_external)
            product of inverse covariance and derivative of mean wrt params
        """
        return tf.einsum(
            "ij,kj->ki",
            Cinv,
            dμ_dθ,
            name="score")

    def get_fisher_matrix(self, Cinv, dμ_dθ, score):
        """Calculate Fisher information matrix

        We calculate the Fisher information matrix. Since we want this to be
        exactly symmetrical to make the inversion quick then we take the lower
        half of the matrix and add it to its transpose.

        Parameters
        __________
        Cinv : TF tensor float (n_summaries, n_summaries) or
                          (n_summaries + n_external, n_summaries + n_external)
            inverse covariance of the summaries
        dμ_dθ : TF tensor float (n_params, n_summaries) or
                                (n_params, n_summaries + n_external)
            derivative mean summaries wrt parameters
        score : TF tensor float (n_params, n_summaries) or
                                (n_params, n_summaries + n_external)
            product of inverse covariance and derivative of mean wrt params

        Returns
        _______
        TF tensor float (n_params, n_params)
            Fisher information matrix
        """
        F = tf.linalg.band_part(
            tf.einsum(
                "ij,kj->ik",
                dμ_dθ,
                score,
                name="half_fisher"),
            0,
            -1,
            name="triangle_fisher")
        return tf.multiply(
            0.5,
            tf.add(
                F,
                tf.transpose(
                    F,
                    perm=[1, 0],
                    name="transposed_fisher"),
                name="double_fisher"),
            name="fisher")

    def get_regularisation(self, C, Cinv):
        CmI = tf.subtract(C, self.identity)
        CinvmI = tf.subtract(Cinv, self.identity)
        regulariser = tf.multiply(
            0.5,
            tf.add(
                tf.square(
                    tf.norm(CmI,
                            ord="fro",
                            axis=(0, 1))),
                tf.square(
                    tf.norm(CinvmI,
                            ord="fro",
                            axis=(0, 1)))),
            name="regulariser")
        return regulariser, CmI

    def get_r(self, regulariser):
        rate = tf.multiply(-self.α, regulariser)
        e_rate = tf.exp(rate)
        r = tf.divide(
                tf.multiply(
                    self.λ,
                    regulariser),
                tf.add(
                    regulariser,
                    e_rate))
        return r, e_rate

    def set_regularisation_strength(self, ϵ, λ):
        self.λ = tf.Variable(λ, dtype=self.dtype, trainable=False,
                             name="strength")
        self.α = -tf.divide(
            tf.math.log(
                tf.add(tf.multiply(tf.subtract(λ, 1.), ϵ),
                       tf.divide(tf.square(ϵ), tf.add(1., ϵ)))),
            ϵ)

    def setup_MLE(self, dataset=True, θ_fid=None, d=None, dd_dθ=None, δθ=None,
                  s=None, ds_dθ=None):
        if dataset:
            self.MLE_θ_fid = self.test_θ_fid
            if self.single_dataset:
                loop = self.dataset
            else:
                loop = zip(self.dataset, self.derivative_dataset)
            for data in loop:
                d, dd_dθ, s, ds_dθ = self.unpack_data(data,
                                                      self.test_use_external)
                self.MLE_F, _, self.MLE_Cinv, self.MLE_dμ_dθ, self.MLE_μ, _ = \
                    self.get_fisher(
                        d, dd_dθ, self.test_numerical, self.test_use_external,
                        s=s, ds_dθ=ds_dθ, δθ=self.test_δθ)
        else:
            if δθ is not None:
                numerical = True
                δθ = tf.Variable(1. / (2. * δθ), dtype=self.dtype)
            else:
                numerical = False
            if s is not None:
                use_external = True
            else:
                use_external = False
            self.MLE_θ_fid = θ_fid
            self.MLE_F, _, self.MLE_Cinv, self.MLE_dμ_dθ, self.MLE_μ, _ = \
                self.get_fisher(
                    d, dd_dθ, numerical, use_external, s=s, ds_dθ=ds_dθ, δθ=δθ)
        self.MLE_Finv = tf.linalg.inv(self.MLE_F)

    def get_MLE(self, d):
        x = self.model(d)
        return tf.add(
            self.MLE_θ_fid,
            tf.einsum(
                "ij,kj->ki",
                self.MLE_Finv,
                tf.einsum(
                    "ij,kj->ki",
                    self.MLE_dμ_dθ,
                    tf.einsum(
                        "ij,kj->ki",
                        self.MLE_Cinv,
                        tf.subtract(x, self.MLE_μ)))))
