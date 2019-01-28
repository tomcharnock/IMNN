"""Information maximising neural network

This module provides the methods necessary to build and train an information
maximising neural network to optimally compress data down to the number of
model parameters.
"""


__version__ = '0.1dev3'
__author__ = "Tom Charnock"


import tensorflow as tf
import numpy as np
import tqdm
from IMNN.utils.utils import load_parameters, check_data, isnotebook, \
    check_amounts, positive_integer, constrained_float


class IMNN():
    """Information maximising neural network

    The information maximising neural network class contains all the functions
    necessary to train a neural network (sequentially) to maximise the Fisher
    information of the summaries of the input data.

    Attributes
    __________
    _FLOATX : typeTF
        32 bit or 64 TensorFlow tensor floats.
    _INTX : typeTF
        32 bit or 64 TensorFlow tensor integers.
    n_s : int
        number of simulations to calculate summary covariance.
    n_p : int
        number of derivatives simulations to calculate derivative of mean.
    n_summaries : int
        number of summaries to compress data to.
    n_params : int
        number of parameters in Fisher information matrix.
    fiducial : list of float
        fiducial parameter values to train IMNN with.
    input_shape : list of int
        shape of the input data.
    filename : {None, str}
        filename for saving and loading network.
    history : dict
        history object for saving training data.
    sess : objTF
        TF session for evaluating graph members.
    saver : objTF
        TF saver for saving the graph to file.
    load_data : bool
        switch for preloading data into TF tensor.
    get_compressor : list of opTF
        set of operations allowing the compression function to be defined.
    store_gradients : list of tensorTF_float
        set of operations allowing gradients to be added to sequentially.
    reset_gradients : list of opTF
        set of operations to reset stored gradient values to zero.
    apply_gradients : opTF
        operation to update all weights and biases to maximise Fisher.
    """

    def __init__(self, parameters):
        """Reads in initialisation dictionary and sets uninitialised parameters
        to None.

        Parameters
        __________
        parameters : dict
            dictionary of initialisation parameters (see Attributes)
            "dtype" : int, optional
            "number of simulations" : int
            "numbef of derivative simulations" : int
            "fiducial" : list
            "number of summaries" : int
            "input shape" : list
            "filename" : str, optional
        loaded_parameters : dict
            dictionary of loaded and checked parameters (see Attributes)
            "_FLOATX" : typeTF
            "_INTX" : typeTF
            "n_s" : int
            "n_p" : int
            "n_summaries" : int
            "n_params" : int
            "fiducial" : list
            "input_shape" : list
            "filename" : {None, str}
        """
        loaded_parameters = load_parameters(parameters)
        self._FLOATX = loaded_parameters["_FLOATX"]
        self._INTX = loaded_parameters["_INTX"]
        self.n_s = loaded_parameters["n_s"]
        self.n_p = loaded_parameters["n_p"]
        self.n_summaries = loaded_parameters["n_summaries"]
        self.n_params = loaded_parameters["n_params"]
        self.fiducial = loaded_parameters["fiducial"]
        self.input_shape = loaded_parameters["input_shape"]
        self.filename = loaded_parameters["filename"]

        self.history = {"det F": np.array([]), "det test F": np.array([])}
        self.sess = None
        self.saver = None
        self.load_data = None
        self.get_compressor = None
        self.store_gradients = None
        self.get_gradients = None
        self.reset_gradients = None
        self.apply_gradients = None

    def begin_session(self):
        """Start TF session with minimal GPU memory usage

        The TF session is started with growth allowed on the GPU to allow for
        most efficient GPU memmory usage. The network is also saved for the
        first time.

        Called from setup(func, optional; bool, optional)

        Parameters
        __________
        config : objTF
            configuration object to set the allowed growth on GPU

        """
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.save_network(first_time=True)

    def reinitialise_session(self):
        """Reset and/or reinitialise all TF variables

        Calling this will set all weights and bias values to randomly
        initialised values, essentially resetting the network.

        """
        self.sess.run(tf.global_variables_initializer())

    def save_network(self, filename=None, first_time=False):
        """Saver function for TF graph

        Saves the graph to file and saves tensor collection names so that they
        can be recovered later. The saver can be called from within train(...)
        in which case filename is None and thus the internal class filename is
        used. It can also be called by the user to a new filename by passing
        the optional filename string.

        Called from begin_session()

        Parameters
        __________
        filename : str, optional
            filename to save the graph to, this should include directories.
        first_time : bool, optional
            switch to define saver for first time or just save graph.
        savefile : str
            variable which can take user inputted or class filename.
        """
        if (self.filename is None and filename is not None):
            savefile = filename
        elif self.filename is not None:
            savefile = self.filename
        else:
            savefile = None
        if savefile is not None:
            print('saving the graph as ' + savefile + '.meta')
            if first_time:
                self.saver = tf.train.Saver()
                self.saver.save(self.sess, "./" + savefile)
                np.savez("./" + savefile + ".npz",
                         store_gradients=[i.name
                                          for i in self.store_gradients],
                         get_gradients=[i.name for i in self.get_gradients],
                         reset_gradients=[i.name
                                          for i in self.reset_gradients],
                         apply_gradients=[self.apply_gradients.name])
            else:
                self.saver.save(self.sess, "./" + savefile,
                                write_meta_graph=False)

    def restore_network(self):
        """Loader function for TF graph

        Load the TF graph from which the the tensor collections can be restored
        using the class attribute filename. The session is started/restarted.

        Parameters
        __________
        config : objTF
            configuration object to set the allowed growth on GPU
        loader : objTF
            TensorFlow object used to restore the graph
        load_arrays : ndarray
            Numpy array of tensor collection names to be reloaded into lists
        """
        if self.filename is not None:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            loader = tf.train.import_meta_graph("./" + self.filename + ".meta")
            loader.restore(self.sess, self.filename)
            load_arrays = np.load("./" + self.filename + ".npz")
            self.get_compressor = [tf.get_default_graph(
                ).get_tensor_by_name("get_fisher:0"),
                                   tf.get_default_graph(
                ).get_tensor_by_name("get_inverse_fisher:0"),
                                   tf.get_default_graph(
                ).get_tensor_by_name("get_mean:0"),
                                   tf.get_default_graph(
                ).get_tensor_by_name("get_compression:0")]
            self.store_gradients = [tf.get_default_graph(
                ).get_tensor_by_name(i)
                for i in load_arrays["store_gradients"]]
            self.get_gradients = [tf.get_default_graph().get_tensor_by_name(i)
                                  for i in load_arrays["get_gradients"]]
            self.reset_gradients = [tf.get_default_graph(
                ).get_tensor_by_name(i)
                for i in load_arrays["reset_gradients"]]
            self.apply_gradients = tf.get_default_graph(
            ).get_operation_by_name(load_arrays["apply_gradients"][0])
        else:
            print("cannot load model since there is no provided filename")

    def setup(self, network=None, load_data=None):
        """Build network, calculate Fisher information and define compression

        This is where the real business is done.

        The TF tensors for the data are defined either as placeholders which
        can be passed the data during training, or as constant (non-trainable)
        variables which can store the data. Stored data is selected by index
        to be passed through the neural network.

        The neural network output summaries are then scattered into a summaries
        (non-trainable) variable which is stored to calculate the Fisher
        information once all of the summaries variable has been filled. The
        covariance of this is calculated and inverted.

        A copy of the derivative of the simulation with respect to the model
        parameters is passed to the derivative of the network outputs with
        respect to the input summaries (non-trainable) variable. The mean of
        this gives us the derivative of the mean of the summaries with respect
        to the model parameters.

        The inverse covariance of the summaries and the derivative of the mean
        of the summaries is then calculated, the upper part of this triangular
        matrix taken and added to its transpose. This removes numerical noise
        derived from taking the inverse of the covariance.

        The summed form of the log determinant of the Fisher matrix is
        calculated so that any large values of the Fisher matrix does not cause
        problems. The derivative of minus the log determinant of the Fisher
        matrix is calculated with respect to the network output summaries to be
        used to update the weights and biases.

        The Fisher information can also be inverted and the computation of the
        compression of any data stored to be able to calculate the maximum
        likelihood estimates without having to pass any data apart from the
        data whose MLE we want to calculate.

        To calculate the gradient backpropagation sequentially we define some
        storage (non-trainable) variables to hold each sequence of weight
        updates and we apply the update sequentially, take the average of this
        update value and apply that to the weights and bias using the inbuilt
        training optimiser.

        Parameters
        __________
        network : func, optional
            a function describing the compression network in the form of a TF
            graph. It takes in a single input tensor (others can be passed via
            lambda functions) and returns a single set of network summaries.
            In principle one should write their own function for the most
            flexibility (note all variables must be in their own variable
            scope), but there is some limited set of network building functions
            available in networks.networks().
        load_data : dict, optional
            the data can be preloaded into the TF graph by passing in a
            dictionary with elements "data" containing the concatenated
            training and validation data (keep note of the indices since they
            are needed in the training function) and "data_d" containing the
            concatenated training and validation derivative of the simulations
            with respect to the parameters. If no data is provided then the
            class attribute load_data is set to False and data must be provided
            during the training.
        lr : tensorTF_float
            a placeholder for the learning rate of the optimiser
        data_ind : tensorTF_int
            index to grab a chosen data slice to pass through the network.
        data_d_ind : tensorTF_int
            index to grab a chosen slice of the derivative of the data to
            calculate the mean derivative.
        stored_data : tensorTF_float
            variable tensor storing the data in the TF graph.
        stored_data_d : tensorTF_float
            variable tensor storing the derivative of the data in the TF graph.
        data : tensorTF_float
            either a placeholder for the data to pass through the network or
            a sliced object in the graph of the data to pass through the
            network.
        data_d : tensorTF_float
            either a placeholder for the derivative of the data to calculate
            the derivative of the mean or a sliced object in the graph of the
            derivative of the data to calculate the derivative of the mean.
        index : tensorTF_int
            placeholder of the index at which to insert the calculated summary
            in the stored summaries variable.
        summaries : tensorTF_float
            variable to store network summaries of simulations so that their
            covariance can be calculated even when the simulations are passed
            sequentially.
        summaries_d : tensorTF_float
            variable to store the derivative of the summaries so that the
            derivative of the mean can be calculated sequentially.
        fiducial : tensorTF_float
            a stored variable containing fiducial parameters for calculating
            the score compression.
        fisher : tensorTF_float
            the stored Fisher information so that the graph doesn't need to be
            run when we want access to it.
        inv_fisher : tensorTF_float
            stored inverse of the Fisher information for use in the score
            compression and for Cramér-Rao estimates.
        mean : tensorTF_float
            stored mean of the summaries for use in the score compression.
        compressor : tensorTF_float
            stored derivative of the mean and inverse covariance vector for
            compression of the summary.
        summary : tensorTF_float
            output tensor from the neural network.
        summary_d : tensorTF_float
            output tensor from the neural network (from a set of summaries used
            only to calculate the gradient of the network with respect to the
            data).
        dxdd : tensorTF_float
            derivative of the network summary with respect to the input data.
        get_summaries : opTF
            operation for placing calculated summaries into indexed storage
            slice for the summaries.
        get_summaries_d : opTF
            operation for placing the calculated derivative of the summaries
            with respect to the model parameters into indexed storage.
        dmudtheta : tensorTF_float
            derivative of the mean with respect to model parameters.
        mu : tensorTF_float
            mean of the network summaries of the simulations.
        diff : tensorTF_float
            difference between the calculated network summary and the mean
            summary values
        cov : tensorTF_float
            covariance of the network summaries of the simulations
        inv_cov : tensorTF_float
            inverse of the covariance of the network summaries of the data
        compression : tensorTF_float
            derivative of the mean and inverse covariance vector for
            compression of the summary.
        onesided_fisher : tensorTF_float
            the Fisher information matrix calculated from the inverse
            covariance of the summaries and the derivative of the mean of the
            summaries with respect to the model parameters.
        calculate_fisher : tensorTF_float
            the Fisher information matrix made exactly symmetric (avoiding
            numerical noise)
        temp_logdetfisher : tensorTF_float
            the sign and value of the log determinant of the Fisher matrix.
        logdetfisher : tensorTF_float
            the log determinant of the Fisher information matrix.
        fisher_gradient : tensorTF_float
            the calculated gradient of the log determinant of the Fisher
            information matrix with respect to the network outputs.
        get_fisher : opTF
            operation to store the value of the Fisher information matrix from
            the stored summaries and derivative of the summaries.
        get_inverse_fisher : opTF
            operation to store the inverse of the Fisher information matrix.
        get_mean : opTF
            operation to store the mean value of the stored network summaries.
        get_compression : opTF
            operation to store the derivative of the mean and inverse
            covariance vector for compression of the summary.
        get_compressor : list of opTF
            a list of operations to run at once to get the score compression.
        MLE : tensorTF_float
            the maximum likelihood estimator from the network.
        trainer : objTF
            the training optimisation scheme.
        gradients : list of tuple of tensorTF_float
            the calculated gradients of the weights and biases of the networks
            for the backpropagation.
        store_gradients : list of tensorTF_float
            a list of tensors with which to store the accumulated gradients.
        get_gradients : list of opTF
            a list of operations to grab the value of the gradients given some
            forward passes of the data.
        reset_gradients : list of opTF
            a list of operations to set the value of the accumulated gradients
            to zero.
        apply_gradients : opTF
            operation to update the weights and biases using the gradient of
            the negative log deterimant of the Fisher information matrix.
        """
        if load_data is not None:
            self.load_data = True
            check_data(self, load_data)
        else:
            self.load_data = False
        lr = tf.placeholder(dtype=self._FLOATX, shape=(), name="learning_rate")
        if self.load_data:
            data_ind = tf.placeholder(dtype=self._INTX, shape=(None, 1),
                                      name="data_ind")
            data_d_ind = tf.placeholder(dtype=self._INTX, shape=(None, 1),
                                        name="data_d_ind")
            stored_data = tf.Variable(load_data["data"], dtype=self._FLOATX,
                                      trainable=False, name="stored_data")
            stored_data_d = tf.Variable(load_data["data_d"],
                                        dtype=self._FLOATX, trainable=False,
                                        name="stored_data_d")
            data = tf.gather_nd(stored_data, data_ind, name="data")
            data_d = tf.gather_nd(stored_data_d, data_d_ind, name="data_d")
        else:
            data = tf.placeholder(dtype=self._FLOATX, shape=[None]
                                  + self.input_shape, name="data")
            data_d = tf.placeholder(dtype=self._FLOATX,
                                    shape=[None, self.n_params]
                                    + self.input_shape, name="data_d")

        index = tf.placeholder(dtype=self._INTX, shape=(None), name="index")

        summaries = tf.Variable(np.zeros((self.n_s, self.n_summaries)),
                                dtype=self._FLOATX, trainable=False,
                                name="summaries")
        summaries_d = tf.Variable(np.zeros((self.n_p, self.n_params,
                                            self.n_summaries)),
                                  dtype=self._FLOATX, trainable=False,
                                  name="summaries_d")

        fiducial = tf.Variable([self.fiducial], dtype=self._FLOATX,
                               trainable=False, name="fiducial")
        fisher = tf.Variable(np.zeros((self.n_params, self.n_params)),
                             dtype=self._FLOATX, trainable=False,
                             name="fisher")
        inv_fisher = tf.Variable(np.zeros((self.n_params, self.n_params)),
                                 dtype=self._FLOATX, trainable=False,
                                 name="inverse_fisher")
        mean = tf.Variable(np.zeros((1, self.n_summaries)), dtype=self._FLOATX,
                           trainable=False, name="saved_mean")
        compressor = tf.Variable(np.zeros((self.n_params, self.n_summaries)),
                                 dtype=self._FLOATX, trainable=False,
                                 name="compressor")

        with tf.variable_scope("IMNN") as scope:
            summary = tf.identity(network(data), name="summary")
            scope.reuse_variables()
            summary_d = tf.split(tf.reshape(network(tf.reshape(data_d, [-1]
                                 + self.input_shape)), [-1, self.n_params,
                                 self.n_summaries], name="summary_d"),
                                 self.n_summaries, axis=2)
            dxdd = tf.transpose(tf.stack([tf.gradients(summary_d[i], data_d)[0]
                                          for i in range(self.n_summaries)]),
                                (1, 2, 0, 3), name="summary_derivative")

        get_summaries = tf.scatter_update(summaries, index, summary,
                                          name="get_summaries")
        get_summaries_d = tf.scatter_update(summaries_d, index,
                                            tf.einsum("ijkl,ijl->ijk", dxdd,
                                                      data_d),
                                            name="get_summaries_d")

        dmudtheta = tf.reduce_mean(summaries_d, axis=0, name="mean_derivative")
        mu = tf.reduce_mean(summaries, axis=0, keepdims=True, name="mean")
        diff = tf.subtract(summaries, mu, name="difference_from_mean")
        cov = tf.divide(tf.einsum("ij,ik->jk", diff, diff), self.n_s - 1.,
                        name="covariance")
        inv_cov = tf.linalg.inv(cov, name="inverse_covariance")
        compression = tf.einsum('ij,kj->ki', inv_cov, dmudtheta,
                                name="compression")
        onesided_fisher = tf.matrix_band_part(tf.einsum('ij,kj->ki', dmudtheta,
                                                        compression),
                                              0, -1, name="onesided_fisher")
        calculate_fisher = tf.identity(tf.multiply(0.5, tf.add(onesided_fisher,
                                       tf.transpose(onesided_fisher))),
                                       name="fisher")
        temp_logdetfisher = tf.linalg.slogdet(calculate_fisher)
        logdetfisher = tf.multiply(temp_logdetfisher[0], temp_logdetfisher[1],
                                   name="logdetfisher")
        fisher_gradient = tf.reduce_sum(tf.gradients(-logdetfisher,
                                                     summaries)[0],
                                        axis=0, keepdims=True,
                                        name="fisher_gradient")

        get_fisher = tf.assign(fisher, calculate_fisher, name="get_fisher")
        get_inverse_fisher = tf.assign(inv_fisher,
                                       tf.linalg.inv(calculate_fisher),
                                       name="get_inverse_fisher")
        get_mean = tf.assign(mean, mu, name="get_mean")
        get_compression = tf.assign(compressor, compression,
                                    name="get_compression")
        self.get_compressor = [get_fisher, get_inverse_fisher, get_mean,
                               get_compression]

        MLE = tf.add(fiducial, tf.einsum("ij,kj->ki", inv_fisher,
                                         tf.einsum("ij,kj->ki", compressor,
                                                   tf.subtract(summary,
                                                               mean))),
                     name="MLE")

        trainer = tf.train.AdamOptimizer(lr)
        gradients = trainer.compute_gradients(input_fisher(summary))
        self.store_gradients = [tf.Variable(np.zeros(gradients[i][0].shape),
                                            dtype=self._FLOATX,
                                            trainable=False,
                                            name="store_gradients_" + str(i))
                                for i in range(len(gradients))]
        self.get_gradients = [tf.assign_add(self.store_gradients[i],
                                            gradients[i][0],
                                            name="add_gradients_" + str(i))
                              for i in range(len(gradients))]
        self.reset_gradients = [tf.assign(self.store_gradients[i],
                                          tf.zeros_like(gradients[i][0]),
                                          name="reset_gradients_" + str(i))
                                for i in range(len(gradients))]
        self.apply_gradients = trainer.apply_gradients([
            (tf.divide(self.store_gradients[i], self.n_s), gradients[i][1])
            for i in range(len(gradients))])
        self.begin_session()

    def train(
            self, updates, at_once, learning_rate, num_sims, num_partial_sims,
            num_validation_sims, num_validation_partial_sims,
            training_dictionary={}, validation_dictionary={}, get_history=True,
            data={"data": None, "data_d": None}, restart=False):
        """Training function for the information maximising neural networks

        This function takes in the data and derivative of the data with respect
        to the model parameters to train the network with. The data can be
        preloaded in the TF graph, in which case the number of simulations and
        number of simulations for validation can be passed to the training
        function to split the stored data into training and validation sets.
        The data can also be passed to the function in a dictionary, again to
        be split into training and validation sets.

        The values of the determinant of the Fisher information for the
        training and validation sets are stored in the history dictionary by
        default.

        If something goes wrong and the network needs to be reinitialised then
        this can be done before retraining.

        You should really use a training set significantly larger than the
        number of simutions needed to calculate the covariance of the summaries
        which will be shuffled through at random. Note that this means that the
        idea of "epochs" does not exist as traditionally though of.

        Parameters
        __________
        updates : int
            number of weight and bias updates to perform.
        at_once: int
            number of simulations to pass through the network at one time.
        learning_rate: float
            the learning rate for the Adam optimiser.
        num_sims : int
            number of simulations to preload to tensor/pass for training.
        num_partial_sims : int
            number of simulations for mean derivative to preload to tensor/
            pass for training.
        num_validation_sims : int
            number of simulations to preload to tensor/pass for validation.
        num_validation_partial_sims : int
            number of simulations for mean derivative to preload to tensor/
            pass for validation.
        training_dictionary : dict, optional
            a dictionary for any tensors which need to be passed to the graph
            during training with custom architectures.
        validation_dictionary : dict, optional
            a dictionary for any tensors which need to be passed to the graph
            during validation when using custom architectures.
        get_history : bool, optional
            a switch to run the validation or not (quicker if not run, but no
            feedback).
        data : dict, optional
            the dictionary which contains the data to train with (if not
            preloaded to the TF graph during setup).
        restart : bool, optional
            a switch whether to reinitialise the network and train from scratch
            or not.
        bar : func
            the function for the progress bar. this must be different depending
            on whether this is run in a notebook or not.
        data_size : int
            the total amount of data, this must be able to be split into number
            of training simulations and number of validation simulations.
        data_d_size : int
            the total amount of derivatives of the data, this must be able to
            be split into the number of training derivatives and validation
            simulations.
        to_test : bool
            if the number of validation simulations is correct then allow
            validation.
        training_ind : ndarray
            array of all possible indices for the training data.
        training_ind_d : ndarray
            array of all possible indices for the training derivatives.
        test_indices : ndarray
            array of all possible indices for the validation data.
        test_indices_d : ndarray
            array of all possible indices for the validation derivatives.
        update_bar : objTQDM
            progress bar for the number of weight and bias updates.
        update : int
            counter for the number of updates
        sim : int
            conuter for the initial simulation of each batch of at_once to be
            passed at once.
        current_dictionary : dict
            dictionary containing the data, or the index of the data, to pass
            through the network and the indices of the summary variables to
            fill.
        test_dictionary : dict
            dictionary containing the validation data, or the index of the
            data, to pass through the network and the indices of the summary
            variables to fill.
        """
        if isnotebook():
            bar = tqdm.tqdm_notebook
        else:
            bar = tqdm.tqdm

        if not self.load_data:
            check_data(self, data)
            data_size = data["data"].shape[0]
            data_d_size = data["data_d"].shape[0]
        else:
            data_size = tf.get_default_graph(
                ).get_tensor_by_name("stored_data:0").get_shape()[0]
	    data_d_size = 
            tf.get_default_graph(
                ).get_tensor_by_name("stored_data_d:0").get_shape()[0]

        to_test = check_amounts(num_sims, num_partial_sims,
                                num_validation_sims,
                                num_validation_partial_sims,
                                data_size, data_d_size)
        updates = positive_integer(updates, key="number of updates")
        at_once = positive_integer(at_once, key="how many simulations to ⁠pass \
            at once")
        learning_rate = constrained_float(learning_rate, key="learning rate")

        training_ind = np.arange(num_sims)
        training_ind_d = np.arange(num_partial_sims)

        if to_test:
            test_indices = np.arange(num_validation_sims) + num_sims
            test_indices_d = np.arange(num_validation_partial_sims)
            + num_partial_sims

        if restart:
            self.history = {"det F": np.array([]),
                            "det test F": np.array([])}
            self.sess.run(tf.global_variables_initializer())

        update_bar = bar(range(updates), desc="Updates")
        for update in update_bar:
            np.random.shuffle(training_ind)
            np.random.shuffle(training_ind_d)
            for sim in range(0, self.n_s, at_once):
                current_dictionary = self.get_dictionary(np.arange(sim, min(sim
                                                                   + at_once,
                                                                   self.n_s)),
                                                         data["data"],
                                                         training_ind,
                                                         derivative=False)
                self.sess.run("get_summaries",
                              feed_dict={**training_dictionary,
                                         **current_dictionary})
            for sim in range(0, self.n_p, at_once):
                current_dictionary = self.get_dictionary(np.arange(sim, min(sim
                                                                   + at_once,
                                                                   self.n_p)),
                                                         data["data_d"],
                                                         training_ind_d,
                                                         derivative=True)
                self.sess.run("get_summaries_d",
                              feed_dict={**training_dictionary,
                                         **current_dictionary})
            self.sess.run(self.reset_gradients)
            for sim in range(0, self.n_s, at_once):
                current_dictionary = self.get_dictionary(np.arange(sim, min(sim
                                                                   + at_once,
                                                                   self.n_s)),
                                                         data["data"],
                                                         training_ind,
                                                         derivative=False)
                self.sess.run(self.get_gradients,
                              feed_dict={**training_dictionary,
                                         **current_dictionary})
            self.sess.run(self.apply_gradients,
                          feed_dict={"learning_rate:0": learning_rate})

            if get_history:
                np.random.shuffle(training_ind)
                np.random.shuffle(training_ind_d)
                for sim in range(0, self.n_s, at_once):
                    current_dictionary = self.get_dictionary(np.arange(sim,
                                                             min(sim + at_once,
                                                                 self.n_s)),
                                                             data["data"],
                                                             training_ind,
                                                             derivative=False)
                    self.sess.run("get_summaries",
                                  feed_dict={**validation_dictionary,
                                             **current_dictionary})
                for sim in range(0, self.n_p, at_once):
                    current_dictionary = self.get_dictionary(np.arange(sim,
                                                             min(sim + at_once,
                                                                 self.n_p)),
                                                             data["data_d"],
                                                             training_ind_d,
                                                             derivative=True)
                    self.sess.run("get_summaries_d",
                                  feed_dict={**validation_dictionary,
                                             **current_dictionary})
                self.history["det F"] = np.concatenate([self.history["det F"],
                                                        [np.exp(self.sess
                                                         .run("logdetfisher:0"
                                                              ))]])

                if to_test:
                    np.random.shuffle(test_indices)
                    np.random.shuffle(test_indices_d)
                    for sim in range(0, self.n_s, at_once):
                        test_dictionary = self.get_dictionary(np.arange(sim,
                                                              min(sim
                                                                  + at_once,
                                                                  self.n_s)
                                                             ),
                                                             data["data"],
                                                             test_indices,
                                                             derivative=False)
                        self.sess.run("get_summaries",
                                      feed_dict={**validation_dictionary,
                                                 **test_dictionary})
                    for sim in range(0, self.n_p, at_once):
                        test_dictionary = self.get_dictionary(np.arange(sim,
                                                              min(sim
                                                                  + at_once,
                                                                  self.n_p)
                                                                 ),
                                                              data["data_d"],
                                                              test_indices_d,
                                                              derivative=True)
                        self.sess.run("get_summaries_d",
                                      feed_dict={**validation_dictionary,
                                                 **test_dictionary})
                    self.history["det test F"] = np.concatenate(
                        [self.history["det test F"], [np.exp(self.sess
                                                      .run("logdetfisher:0"))]]
                                                      )
                    update_bar.set_postfix(F=self.history["det F"][-1],
                                           validation_F=self
                                           .history["det test F"][-1])
                else:
                    update_bar.set_postfix(F=self.history["det F"][-1])

    def get_dictionary(self, inds, data, indices, derivative=None):
        """Get the dictionary to pass objects into the TF graph

        Function fills a dictionary with slices of data to be passed to the
        graph if the data is not preloaded to the graph, otherwise it reshapes
        the indices to slice through the tensors already in the graph.

        This dictionary changes on every pass of the data which is why it is
        update here.

        The index of the stored variables is also denoted here.

        We pass the objects to the tensors by name so that we don't have to
        pass around tensor objects outside of the graph by name.

        Parameters
        __________
        inds : ndarray
            the values of the indices to pass at which to fill the stored
            summary values
        data : ndarray
            the value of the data to be sliced and passed to the network.
            if the data is preloaded to the network then the data variable is
            overwritten with the values of the indices to slice the stored data
            at.
        indices : ndarray
            the random index array to randomly select data to pass to the
            network, or the random indices to pass to the network to select the
            data if it is already preloaded.
        derivative : bool, optional
            a switch to choose the name of the tensor to send the data to in
            the feed dictionary.
        key : str
            the name of the tensor in the TF graph to fill to calculate the
            network output.
        dictionary : dict
            the holder dictionary to be returned with the data and the indices.

        Returns
        _______
        dict
            the holder dictionary to be returned with the data and the indices.
        """
        if self.load_data:
            data = indices[inds][:, np.newaxis]
            if derivative:
                key = "data_d_ind:0"
            else:
                key = "data_ind:0"
        else:
            data = data[indices[inds]]
            if derivative:
                key = "data_d:0"
            else:
                key = "data:0"

        dictionary = {}
        dictionary["index:0"] = inds
        dictionary[key] = data
        return dictionary


@tf.custom_gradient
def input_fisher(x):
    """
    Self defined gradient for sequential backpropagation.

    This must be defined outside of classes, but it is not called until
    IMNN.IMNN().train().

    Parameters
    __________
    x : tensorTF_float
        IMNN summaries (output of the neural network)

    Returns
    _______
    tensorTF_float
        an identity operator on the IMNN summaries
    tensorTF_float
        a tiled tensor with the shape of the gradients of all weights and
        biases, filled with the self defined tensor living in the TF graph.
    """
    def grad(dy):
        """
        Gradient of the log determinant of the Fisher information

        Grabs the tensor from the TF graph and tiles it to the correct shape.

        Parameters
        __________
        dy : tensorTF_float
            inputted tensor to be update during the backpropagation.
        gradient : tensorTF_float
            the derivative of the log determinant of the Fisher matrix from
            TF graph.
        num_gradients : int
            number of inidividual gradients to tile correct shape tensor

        Returns
        _______
        tensorTF_float
            tiled gradient with shape of input gradient tensor

        """
        gradient = tf.get_default_graph().get_tensor_by_name(
            "fisher_gradient:0")
        num_gradients = len(gradient.get_shape().as_list()) - 1
        return tf.tile(
            gradient, [tf.shape(dy)[0]] + [1 for i in range(num_gradients)])
    return tf.identity(x), grad
