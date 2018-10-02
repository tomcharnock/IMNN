import tensorflow as tf
import numpy as np
import tqdm
import utils

class IMNN():
    def __init__(n, parameters):
        # INITIALISE NETWORK PARAMETERS
        #______________________________________________________________
        # INPUTS
        # parameters                    dict      - dictionary containing initialisation parameters
        #______________________________________________________________
        # FUNCTIONS (DEFINED IN utils.py)
        # check_params(dict)                      - checks that all necessary parameters are in dictionary
        # isboolean(list)               bool      - checks that parameter is a boolean
        # inputs(dict)                  list      - checks shape of network input
        # positive_integer(list)        int       - checks that parameter is a positive integer
        # number_of_derivative_simulations(dict, class)
        #                                         - calculates the number of simulations to use for numerical derivative
        # check_preloaded(dict, class)  dict/None - checks preloaded data or returns None if not preloading
        # check_fiducial(dict)          array,int - checks fiducial parameter values are passed correctly and gets number of parameters
        # check_derivative(dict, class) array     - checks derivative denominator shape
        # check_save_file(dict)         str/None  - checks that file name is a string or None
        # isfloat(list)                 float     - checks that parameter is a float
        # check_prebuild_params(dict)             - checks that all parameters for prebuilt network are in dictionary
        # auto_initialise(float)        bool      - switch to choose whether to use constant weight variance
        # activation(dict)              tf func   - checks that TF activation function is allowed
        #                               bool
        #                               float/int
        # hidden_layers(dict, class)    list      - checks that hidden layers can be built into a network
        # initialise_variables()        21 Nones  - sets all other class variables to None
        #______________________________________________________________
        # VARIABLES
        # u                             class     - utility functions
        # _FLOATX                     n tf type   - set TensorFlow types to 32 bit (for GPU)
        # verbose                     n bool      - True to print outputs such as shape of tensors
        # inputs                      n list      - shape of input (list)
        # n_s                         n int       - total number of simulations
        # n_p                         n int       - number of differentiation simulations
        # preload_data                n dict/None - training data to be preloaded to TensorFlow constant
        # fiducial_θ                  n array     - fiducial parameter values to be loaded as a TensorFlow constant
        # derivative_denominator      n array     - derivative denominator values to be loaded as TensorFlow constant
        # n_params                    n int       - number of parameters in the model
        # n_summaries                 n int       - number of outputs from the network
        # prebuild                    n bool      - True to allow IMNN to build the network
        # save_file                   n str/None  - Name to save or load graph. None does not save graph
        # get_MLE                     n bool      - True to calculate MLE
        # wv                          n float     - weight variance to initialise all weights
        # allow_init                  n bool      - True if nw < 0 so network calculates weight variance
        # bb                          n float     - constant bias initialiser value
        # activation                  n tf func   - activation function to use
        # takes_α                     n bool      - True is an extra parameter is needed in the activation function
        # α                           n float/int - extra parameter if needed for activation function (can be None)
        # layers                      n list      - contains the neural architecture of the network
        # sess        begin_session() n session   - interactive tensorflow session (with initialised parameters)
        # x                   setup() n tensor    - fiducial simulation input tensor
        # x_central           setup() n tensor    - input tensor for fiducial simulations
        # central_indices     setup() n tensor    - list of indices to select preloaded central data at
        # derivative_indices  setup() n tensor    - list of indices to select preloaded derivative data at
        # θ_fid               setup() n tensor    - fiducial parameter input tensor for calculation MLE
        # prior               setup() n tensor    - prior range for each parameter for calculating MLE
        # x_m                 setup() n tensor    - below fiducial simulation input tensor
        # x_p                 setup() n tensor    - above fiducial simulation input tensor
        # dd                  setup() n tensor    - inverse difference between upper and lower parameter value
        # dropout             setup() n tensor    - keep rate for dropout layer
        # output              setup() n tensor    - network output for simulations at fiducial parameter value
        # F                   setup() n tensor    - Fisher information matrix
        # μ                   setup() n tensor    - mean of network outputs for fiducial simulations
        # C                   setup() n tensor    - covariance of network outputs for fiducial simulations
        # iC                  setup() n tensor    - inverse covariance of network outputs for fiducial simulations
        # dμdθ                setup() n tensor    - numerical derivative of the mean of network outputs
        # Λ                   setup() n tensor    - value of the loss function
        # test_F              setup() n tensor    - Fisher information from test data
        # test_μ              setup() n tensor    - mean of network outputs for fiducial test simulations
        # test_C              setup() n tensor    - covariance of network outputs for fiducial test simulations
        # test_iC             setup() n tensor    - inverse covariance of network outputs for fiducial test simulations
        # test_dμdθ           setup() n tensor    - numerical derivative of the mean of network outputs from test simulations
        # test_Λ              setup() n tensor    - loss function from test simulations
        # MLE                 setup() n tensor    - MLE of parameters
        # AL                  setup() n tensor    - asymptotic likelihood at range of parameter values
        # backpropagate       setup() n tf opt    - minimisation scheme for the network
        # saver        save_network() n tf op     - saver operation for saving model
        # history             train() n dict      - dictionary for training history
        #______________________________________________________________
        u = utils.utils()
        n._FLOATX = tf.float32
        u.check_params(parameters)
        n.verbose = u.isboolean([parameters, 'verbose'])
        n.inputs = u.inputs(parameters)
        n.n_s = u.positive_integer([parameters, 'number of simulations'])
        n.n_p = u.number_of_derivative_simulations(parameters, n)
        n.fiducial_θ, n.n_params = u.check_fiducial(parameters)
        n.derivative_denominator = u.check_derivative(parameters, n)
        n.preload_data = u.check_preloaded(parameters, n)
        n.n_summaries = u.positive_integer([parameters, 'number of summaries'])
        n.prebuild = u.isboolean([parameters, 'prebuild'])
        n.save_file = u.check_save_file([parameters, 'save file'])
        n.get_MLE = u.isboolean([parameters, 'calculate MLE'])
        if n.prebuild:
            u.check_prebuild_params(parameters)
            n.wv = u.isfloat([parameters, 'wv'])
            n.allow_init = u.auto_initialise(n.wv)
            n.bb = u.isfloat([parameters, 'bb'])
            n.activation, n.takes_α, n.α = u.activation(parameters)
            n.layers = u.hidden_layers(parameters, n)
        n.sess, n.x, n.x_central, n.central_indices, n.derivative_indices, n.θ_fid, n.prior, n.x_m, n.x_p, n.dd, n.dropout, n.output, n.F, n.μ, n.C, n.iC, n.dμdθ, n.Λ, n.test_F, n.test_μ, n.test_C, n.test_iC, n.test_dμdθ, n.test_Λ, n.MLE, n.AL, n.backpropagate, n.saver, n.history = u.initialise_variables()

    def begin_session(n):
        # BEGIN TENSORFLOW SESSION AND INITIALISE PARAMETERS
        #______________________________________________________________
        # CALLED FROM (DEFINED IN IMNN.py)
        # setup(optional func)                    - builds generic or auto built network
        # save_network(optional Bool)             - saves the network (if n.save_file is not None)
        #______________________________________________________________
        # VARIABLES
        # config                                  - tensorflow GPU configuration options
        # sess                        n session   - tensorflow session (with initialised parameters)
        #______________________________________________________________
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        n.sess = tf.Session(config = config)
        n.sess.run(tf.global_variables_initializer())
        n.save_network(first_time = True)

    def reinitialise_session(n):
        # REINITIALISE PARAMETERS
        #______________________________________________________________
        # CALLED FROM (DEFINED IN IMNN.py)
        # restore_network(string, float)          - restores network from a tensorflow saver metafile
        #______________________________________________________________
        # VARIABLES
        # sess                        n session   - interactive tensorflow session
        #______________________________________________________________
        n.sess.run(tf.global_variables_initializer())

    def save_network(n, file_name = None, first_time = False):
        # SAVE NETWORK AS A TENSORFLOW SAVER METAFILE
        #______________________________________________________________
        # INPUTS
        # file_name                     str/None  - name of the file to reload (ending in .meta)
        # first_time                    bool      - whether to save the meta graph or not
        # save_file                   n str/None  - name of the file to reload (ending in .meta)
        #______________________________________________________________
        # VARIABLES
        # save_file                     str/None  - Name to save or load graph. None does not save graph
        # saver                       n tf op     - saving operation from tensorflow
        #______________________________________________________________
        if (n.save_file is None and file_name is not None):
            save_file = file_name
        elif n.save_file is not None:
            save_file = n.save_file
        else:
            save_file = None
        if save_file is not None:
            print('saving the graph as ' + save_file + '.meta')
            if first_time:
                n.saver = tf.train.Saver()
                n.saver.save(n.sess, "./" + save_file)
            else:
                n.saver.save(n.sess, "./" + save_file, write_meta_graph = False)

    def restore_network(n):
        # RESTORES NETWORK FROM TENSORFLOW SAVER METAFILE
        #______________________________________________________________
        # INPUTS
        # save_file                   n string    - name of the file to reload (ending in .meta)
        #______________________________________________________________
        # VARIABLES
        # config                                  - tensorflow GPU configuration options
        # sess                        n session   - interactive tensorflow session (with initialised parameters)
        # loader                        tf op     - loading operation from tensorflow
        # x                           n tensor    - fiducial simulation input tensor
        # x_central                   n tensor    - fixed-sized fiducial simulation input tensor
        # x_m                         n tensor    - below fiducial simulation input tensor
        # x_p                         n tensor    - above fiducial simulation input tensor
        # dd                          n tensor    - inverse difference between upper and lower parameter value
        # dropout                     n tensor    - keep rate for dropout layer
        # output                      n tensor    - network output for simulations at fiducial parameter value
        # F                           n tensor    - Fisher information matrix
        # C                           n tensor    - covariance of network outputs for fiducial simulations
        # iC                          n tensor    - inverse covariance of network outputs for fiducial simulations
        # μ                           n tensor    - mean of network outputs for fiducial simulations
        # dμdθ                        n tensor    - numerical derivative of the mean of network outputs
        # Λ                           n tensor    - loss function
        # test_F                      n tensor    - Fisher information matrix from test simulations
        # test_iC                     n tensor    - inverse covariance of network outputs for fiducial test simulations
        # test_μ                      n tensor    - mean of network outputs for fiducial test simulations
        # test_dμdθ                   n tensor    - numerical derivative of the mean of network outputs from test simulations
        # test_C                      n tensor    - covariance of network outputs for fiducial test simulations
        # test_Λ                      n tensor    - loss function from test simulations
        # MLE                         n tensor    - MLE of parameters
        # AL                          n tensor    - asymptotic likelihood at range of parameter values
        # central_indices             n tensor    - list of indices to select preloaded central data at
        # derivative_indices          n tensor    - list of indices to select preloaded derivative data at
        # backpropagate               n tf opt    - minimisation scheme for the network
        #______________________________________________________________
        if n.save_file is not None:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            n.sess = tf.Session(config = config)
            loader = tf.train.import_meta_graph("./" + n.save_file + ".meta")
            loader.restore(n.sess, n.save_file)
            n.x = tf.get_default_graph().get_tensor_by_name("x:0")
            n.x_central = tf.get_default_graph().get_tensor_by_name("x_central:0")
            n.x_m = tf.get_default_graph().get_tensor_by_name("x_m:0")
            n.x_p = tf.get_default_graph().get_tensor_by_name("x_p:0")
            n.θ_fid = tf.get_default_graph().get_tensor_by_name("fiducial:0")
            n.prior = tf.get_default_graph().get_tensor_by_name("prior:0")
            n.dd = tf.get_default_graph().get_tensor_by_name("dd:0")
            n.dropout = tf.get_default_graph().get_tensor_by_name("dropout:0")
            n.output = tf.get_default_graph().get_tensor_by_name("output:0")
            n.F = tf.get_default_graph().get_tensor_by_name("fisher_information:0")
            n.C = tf.get_default_graph().get_tensor_by_name("covariance:0")
            n.iC = tf.get_default_graph().get_tensor_by_name("inverse_covariance:0")
            n.μ = tf.get_default_graph().get_tensor_by_name("mean:0")
            n.dμdθ = tf.get_default_graph().get_tensor_by_name("mean_derivative:0")
            n.Λ = tf.get_default_graph().get_tensor_by_name("loss:0")
            n.test_F = tf.get_default_graph().get_tensor_by_name("test_F:0")
            n.test_C = tf.get_default_graph().get_tensor_by_name("test_covariance:0")
            n.test_iC = tf.get_default_graph().get_tensor_by_name("test_inverse_covariance:0")
            n.test_μ = tf.get_default_graph().get_tensor_by_name("test_mean:0")
            n.test_dμdθ = tf.get_default_graph().get_tensor_by_name("test_mean_derivative:0")
            n.test_Λ = tf.get_default_graph().get_tensor_by_name("test_loss:0")
            if n.get_MLE:
                n.MLE = tf.get_default_graph().get_tensor_by_name("maximum_likelihood_estimate:0")
                n.AL = tf.get_default_graph().get_tensor_by_name("asymptotic_likelihood:0")
            if "central_indices" in [v.name for v in n.sess.graph.get_operations()]:
                n.central_indices = tf.get_default_graph().get_tensor_by_name("central_indices:0")
                n.derivative_indices = tf.get_default_graph().get_tensor_by_name("derivative_indices:0")
            n.backpropagate = tf.get_default_graph().get_operation_by_name("Adam")

    def dense(n, input_tensor, l, dropout):
        # DENSE LAYER
        #______________________________________________________________
        # CALLED FROM (DEFINED IN IMNN.py)
        # build_network(tensor, tensor) tensor    - auto builds predefined network
        #______________________________________________________________
        # RETURNS
        # tensor
        # activated dense output (with dropout)
        #______________________________________________________________
        # INPUTS
        # input_tensor                  tensor    - input tensor to the dense layer
        # l                             int       - layer counter
        # dropout                       tensor    - keep rate for dropout layer
        # layers                      n list      - contains the neural architecture of the network
        # allow_init                  n bool      - True is nw = 0. so network calculates weight variance
        # wv                          n float     - weight variance to initialise the weights
        # bb                          n float     - bias initialiser value
        # n.take_α                    n bool      - True is an extra parameter is needed in the activation function
        # activation                  n tf func   - activation function to use
        # n.α                         n float/int - extra parameter if needed for activation function (can be None)
        #______________________________________________________________
        # VARIABLES
        # previous_layer                int       - shape of previous layer
        # weight_shape                  tuple     - shape of the weight kernel
        # bias_shape                    tuple     - shape of the bias kernel
        # weights                       tensor    - weight kernel
        # biases                        tensor    - bias kernel
        # dense                         tensor    - dense layer (not activated)
        #______________________________________________________________
        previous_layer = int(input_tensor.get_shape().as_list()[-1])
        weight_shape = (previous_layer, n.layers[l])
        bias_shape = (n.layers[l])
        if n.allow_init:
            n.wv = np.sqrt(2. / previous_layer)
        weights = tf.get_variable("weights", weight_shape, initializer = tf.random_normal_initializer(0., n.wv))
        biases = tf.get_variable("biases", bias_shape, initializer = tf.constant_initializer(n.bb))
        dense = tf.add(tf.matmul(input_tensor, weights), biases)
        if n.takes_α:
            return tf.nn.dropout(n.activation(dense, n.α), dropout, name = 'dense_' + str(l))
        else:
            return tf.nn.dropout(n.activation(dense), dropout, name = 'dense_' + str(l))

    def conv(n, input_tensor, l, dropout):
        # CONVOLUTIONAL LAYER
        #______________________________________________________________
        # CALLED FROM (DEFINED IN IMNN.py)
        # build_network(tensor, tensor) tensor    - auto builds predefined network
        #______________________________________________________________
        # RETURNS
        # tensor
        # activated convolutional output (with dropout)
        #______________________________________________________________
        # INPUTS
        # input_tensor                  tensor    - input tensor to the convolutional layer
        # l                             int       - layer counter
        # dropout                       tensor    - keep rate for dropout layer
        # layers                      n list      - contains the neural architecture of the network
        # allow_init                  n bool      - True is nw = 0. so network calculates weight variance
        # wv                          n float     - weight variance to initialise the weights
        # bb                          n float     - bias initialiser value
        # n.take_α                    n bool      - True is an extra parameter is needed in the activation function
        # activation                  n tf func   - activation function to use
        # n.α                         n float/int - extra parameter if needed for activation function (can be None)
        #______________________________________________________________
        # VARIABLES
        # previous_layer                int       - shape of previous layer
        # weight_shape                  tuple     - shape of the weight kernel
        # bias_shape                    tuple     - shape of the bias kernel
        # weights                       tensor    - weight kernel
        # biases                        tensor    - bias kernel
        # conv                          tensor    - convolutional feature map (not activated)
        #______________________________________________________________
        previous_filters = int(input_tensor.get_shape().as_list()[-1])
        if len(n.layers[l][1]) == 2:
            convolution = tf.nn.conv2d
            weight_shape = (n.layers[l][1][0], n.layers[l][1][1], previous_filters, n.layers[l][0])
        else:
            convolution = tf.nn.conv3d
            weight_shape = (n.layers[l][1][0], n.layers[l][1][1], n.layers[l][1][1], previous_filters, n.layers[l][0])
        bias_shape = (n.layers[l][0])
        if n.allow_init:
            n.wv = np.sqrt(2. / previous_filters)
        weights = tf.get_variable("weights", weight_shape, initializer = tf.random_normal_initializer(0., n.wv))
        biases = tf.get_variable("biases", bias_shape, initializer = tf.constant_initializer(n.bb))
        conv = tf.add(convolution(input_tensor, weights, [1] + n.layers[l][2] + [1], padding = n.layers[l][3]), biases)
        if n.takes_α:
            return tf.nn.dropout(n.activation(conv, n.α), dropout, name = 'conv_' + str(l))
        else:
            return tf.nn.dropout(n.activation(conv), dropout, name = 'conv_' + str(l))

    def build_network(n, input_tensor, dropout):
        # AUTO BUILD NETWORK
        #______________________________________________________________
        # CALLED FROM (DEFINED IN IMNN.py)
        # setup(optional func)                    - builds generic or auto built network
        #______________________________________________________________
        # RETURNS
        # tensor
        # last tensor of the IMNN architecture (summarised output)
        #______________________________________________________________
        # FUNCTIONS (DEFINED IN IMNN.py)
        # dense(tensor, int, tensor)    tensor    - calculates a dense layer
        # conv(tensor, int, tensor)     tensor    - calculates a convolutional layer
        #______________________________________________________________
        # INPUTS
        # input_tensor                  tensor    - input tensor to the network
        # dropout                       tensor    - keep rate for dropout layer
        # verbose                     n bool      - True to print outputs such as shape of tensors
        # layers                      n list      - contains the neural architecture of the network
        #______________________________________________________________
        # VARIABLES
        # l                             int       - layer counter
        # layer                         list      - list of tensors for each layer of the network
        # drop_val                      tensor    - keep value for dropout layer (set to 1 for output layer)
        #______________________________________________________________
        if n.verbose: print(input_tensor)
        layer = [input_tensor]
        for l in range(1, len(n.layers)):
            if l < len(n.layers) - 1:
                drop_val = dropout
            else:
                drop_val = 1.
            if type(n.layers[l]) == list:
                if len(layer[-1].get_shape().as_list()) < 2:
                    layer.append(tf.reshape(layer[-1], (-1, layer[-1].get_shape().as_list()[-1], 1)))
                with tf.variable_scope('layer_' + str(l)):
                    layer.append(n.conv(layer[-1], l, drop_val))
            else:
                if len(layer[-1].get_shape()) > 2:
                    layer.append(tf.reshape(layer[-1], (-1, np.prod(layer[-1].get_shape().as_list()[1:]))))
                with tf.variable_scope('layer_' + str(l)):
                    layer.append(n.dense(layer[-1], l, drop_val))
            if n.verbose: print(layer[-1])
        return layer[-1]

    def inverse_covariance(n, a):
        # CALCULATE THE INVERSE COVARIANCE, MEAN AND COVARIANCE
        #______________________________________________________________
        # CALLED FROM (DEFINED IN IMNN.py)
        # Fisher(tensor, tensor, tensor, tensor)
        #                               tensor    - builds generic or auto built network
        #______________________________________________________________
        # RETURNS
        # tensor, tensor, tensor
        # inverse covariance, mean, covariance
        #______________________________________________________________
        # INPUTS
        # a                             tensor    - network output for simulations at fiducial parameter value
        # n_s                         n int       - number of simulations in each combination
        # n_summaries                 n int       - number of outputs from the network
        # verbose                     n bool      - True to print outputs such as shape of tensors
        #______________________________________________________________
        # VARIABLES
        # a_                            tensor    - reshaped network outputs for fiducial simulations
        # μ                           n tensor    - mean of network outputs for fiducial simulations
        # outmm                         tensor    - difference between mean and network outputs for fiducial simulations
        # C                           n tensor    - covariance of network outputs for fiducial simulations
        # iC                          n tensor    - inverse covariance of network outputs for fiducial simulations
        #______________________________________________________________
        a_ = tf.reshape(a, (n.n_s, n.n_summaries), name = 'central_output')
        if n.verbose: print(a_)
        μ = tf.reduce_mean(a_, axis = 0, name = 'central_mean')
        if n.verbose: print(μ)
        outmm = tf.subtract(a_, tf.expand_dims(μ, 0), name = 'central_difference_from_mean')
        if n.verbose: print(outmm)
        C = tf.divide(tf.einsum('ij,ik->jk', outmm, outmm), tf.constant((n.n_s - 1.), dtype = n._FLOATX), name = 'central_covariance')
        if n.verbose: print(C)
        iC = tf.matrix_inverse(C, name = 'inverse_central_covariance')
        if n.verbose: print(iC)
        return iC, μ, C

    def derivative_mean(n, a_m, a_p):
        # CALCULATE THE DERIVATIVE OF THE MEAN OF THE SIMULATIONS
        #______________________________________________________________
        # CALLED FROM (DEFINED IN IMNN.py)
        # Fisher(tensor, tensor, tensor, tensor)
        #                               tensor    - builds generic or auto built network
        #______________________________________________________________
        # RETURNS
        # tensor
        # Derivative of the mean of the simulations
        #______________________________________________________________
        # INPUTS
        # a_m                           tensor    - network output for simulations below fiducial parameter value
        # a_p                           tensor    - network output for simulations above fiducial parameter value
        # dd                            tensor    - inverse difference between upper and lower parameter value
        # n_summaries                 n int       - number of outputs from the network
        # verbose                     n bool      - True to print outputs such as shape of tensors
        # n_p                         n int       - number of differentiation simulations in each combination
        #______________________________________________________________
        # VARIABLES
        # a_m_                          tensor    - reshaped network outputs for lower parameter simulations
        # a_p_                          tensor    - reshaped network outputs for upper parameter simulations
        # dμdθ                        n tensor    - numerical derivative of the mean of network outputs
        #______________________________________________________________
        a_m_ = tf.reshape(a_m, (n.n_p, n.n_params, n.n_summaries), name = 'lower_output')
        if n.verbose: print(a_m_)
        a_p_ = tf.reshape(a_p, (n.n_p, n.n_params, n.n_summaries), name = 'upper_output')
        if n.verbose: print(a_p_)
        dμdθ = tf.divide(tf.einsum('ijk,j->jk', tf.subtract(a_p_, a_m_), n.dd), tf.constant(n.n_p, dtype = n._FLOATX), name = 'upper_lower_mean_derivative')
        if n.verbose: print(dμdθ)
        return dμdθ

    def Fisher(n, a, a_m, a_p):
        # CALCULATE THE FISHER INFORMATION
        #______________________________________________________________
        # CALLED FROM (DEFINED IN IMNN.py)
        # setup(optional func)                    - builds generic or auto built network
        #______________________________________________________________
        # RETURNS
        # tensor, tensor, tensor, tensor, tensor
        # Fisher information matrix, inverse covariance, mean, derivative of the mean, covariance
        #______________________________________________________________
        # FUNCTIONS (DEFINED IN IMNN.py)
        # inverse_covariance(tensor)
        #               tensor, tensor, tensor    - calculates inverse covariance, mean and covariance
        # derivative_mean(tensor, tensor)
        #                               tensor    - calculates a derivative of the mean
        #______________________________________________________________
        # INPUTS
        # a                             tensor    - network output for simulations at fiducial parameter value
        # a_m                           tensor    - network output for simulations below fiducial parameter value
        # a_p                           tensor    - network output for simulations above fiducial parameter value
        #______________________________________________________________
        # VARIABLES
        # μ                             tensor    - mean of network outputs for fiducial simulations
        # C                             tensor    - covariance of network outputs for fiducial simulations
        # iC                            tensor    - inverse covariance of network outputs for fiducial simulations
        # dμdθ                          tensor    - numerical derivative of the mean of network outputs
        # F                             tensor    - Fisher information matrix
        #______________________________________________________________
        iC, μ, C = n.inverse_covariance(a)
        dμdθ = n.derivative_mean(a_m, a_p)
        F = tf.matrix_band_part(tf.einsum('ij,kj->ik', dμdθ, tf.einsum('ij,kj->ki', iC, dμdθ)), 0, -1)
        F = 0.5 * (F + tf.transpose(F))
        return F, iC, μ, dμdθ, C

    def calculate_asymptotic_likelihood(n):
        # CALCULATE THE ASYMPTOTIC LIKELIHOOD
        #______________________________________________________________
        # CALLED FROM (DEFINED IN IMNN.py)
        # setup(optional func)                    - builds generic or auto built network
        #______________________________________________________________
        # RETURNS
        # tensor
        # Asymptotic likelihood at a range of parameter values
        #______________________________________________________________
        # INPUTS
        # output                      n tensor    - IMNN summary of real data
        # θ_fid                       n tensor    - fiducial parameter input tensor for calculating MLE
        # prior                       n tensor    - prior range for each parameter for calculating MLE
        # μ                           n tensor    - mean of network outputs for fiducial simulations
        # iC                          n tensor    - inverse covariance of network outputs for fiducial simulations
        # dμdθ                        n tensor    - numerical derivative of the mean of network outputs
        #______________________________________________________________
        # VARIABLES
        # Δθ                            tensor    - range over which to find the maximum of the likelihood
        # approximate_likelihood        tensor    - approximate likelihood over prior range
        #______________________________________________________________
        Δθ = n.prior - tf.expand_dims(tf.expand_dims(n.θ_fid, 0), 2)
        asymptotic_likelihood = tf.exp(- 0.5 * (tf.expand_dims(tf.expand_dims(tf.einsum("ij,ij->i", n.output, tf.einsum("ij,kj->ki", n.test_iC, n.output)) + tf.einsum("i,ji->j", n.test_μ, tf.einsum("ij,kj->ki", n.test_iC, n.output)) + tf.einsum("ij,j->i", n.output, tf.einsum("ij,j->i", n.test_iC, n.test_μ)) + tf.einsum("i,i->", n.test_μ, tf.einsum("ij,j->i", n.test_iC, n.test_μ)), 1), 2) - tf.einsum("ijk,ij->ijk", Δθ, tf.einsum("ij,kj->ki", n.test_dμdθ, tf.einsum("ij,kj->ki", n.test_iC, n.output))) - tf.einsum("ijk,ij->ijk", Δθ, tf.einsum("ij,kj->ik", n.output, tf.einsum("ij,kj->ki", n.test_iC, n.test_dμdθ))) + tf.einsum("ijk,j->ijk", Δθ, tf.einsum("ij,j->i", n.test_dμdθ, tf.einsum("ij,j->i", n.test_iC, n.test_μ))) + tf.einsum("ijk,j->ijk", Δθ, tf.einsum("i,ji->j", n.test_μ, tf.einsum("ij,kj->ki", n.test_iC, n.test_dμdθ))) + tf.einsum("ijk,ijk->ijk", Δθ, tf.einsum("ijk,j->ijk", Δθ, tf.einsum("ij,ij->i", n.test_dμdθ, tf.einsum("ij,kj->ki", n.test_iC, n.test_dμdθ))))))
        return asymptotic_likelihood / tf.reduce_sum(tf.einsum("ijk,ij->ijk", asymptotic_likelihood, (n.prior[:, :, 1] - n.prior[:, :, 0])), 2, keepdims = True)

    def maximum_likelihood_estimate(n):
        # CALCULATE THE MAXIMUM LIKELIHOOD ESTIMATE
        #______________________________________________________________
        # CALLED FROM (DEFINED IN IMNN.py)
        # setup(optional func)                    - builds generic or auto built network
        #______________________________________________________________
        # RETURNS
        # tensor
        # Maximum likelihood estimate
        #______________________________________________________________
        # INPUTS
        # F                           n tensor    - Fisher information matrix from simulations
        # output                      n tensor    - network summary of real data
        # θ_fid                       n tensor    - fiducial parameter input tensor for calculating MLE
        # μ                           n tensor    - mean of network outputs for fiducial simulations
        # Ci                          n tensor    - inverse covariance of network outputs for fiducial simulations
        # dμdθ                        n tensor    - numerical derivative of the mean of network outputs
        #______________________________________________________________
        # VARIABLES
        # iF                            tensor    - inverse Fisher information matrix from fiducial simulations
        #______________________________________________________________
        iF = tf.matrix_inverse(n.test_F)
        return tf.expand_dims(n.θ_fid, 0) + tf.einsum("ij,kj->ki", iF, tf.einsum("ij,kj->ki", tf.einsum("ij,kj->ki", n.test_iC, n.test_dμdθ), n.output - tf.expand_dims(n.test_μ, 0)))

    def loss(n, F):
        # CALCULATE THE LOSS FUNCTION
        #______________________________________________________________
        # CALLED FROM (DEFINED IN IMNN.py)
        # setup(optional func)                    - builds generic or auto built network
        #______________________________________________________________
        # RETURNS
        # tensor
        # loss function (-0.5 * |F|^2)
        #______________________________________________________________
        # INPUTS
        # F                           n tensor    - Fisher information matrix
        # _FLOATX                     n tf type   - TensorFlow type
        #______________________________________________________________
        # VARIABLES
        # IFI                           tensor    - determinant of the Fisher information matrix
        #______________________________________________________________
        IFI = tf.matrix_determinant(F)
        return tf.multiply(tf.constant(-0.5, dtype = n._FLOATX), tf.square(IFI))

    def setup(n, η, network = None):
        # SETS UP GENERIC NETWORK
        #______________________________________________________________
        # FUNCTIONS (DEFINED IN IMNN.py)
        # build_network(tensor, tensor) tensor    - builds predefined network architecture
        # Fisher(tensor, tensor, tensor, tensor)
        #                               tensor    - calculates Fisher information
        # maximumlikelihoodestimate(tensor)
        #                               tensor    - calculates maximum likelihood estimate
        # asymptotic_likelihood(tensor) tensor    - calculates the asymptotic likelihood
        # loss(tensor)                  tensor    - calculates loss function
        # training_scheme(float)        float     - defines the minimisation scheme for backpropagation
        # begin_session()                         - starts interactive tensorflow session and initialises variables
        #______________________________________________________________
        # FUNCTIONS (DEFINED IN utils.py)
        # to_prebuild(func)                       - checks whether network builder is provided
        #______________________________________________________________
        # INPUTS
        # η                             float     - learning rate
        # network              optional func      - externally provided function for building network
        # inputs                      n int/list  - number of inputs (int) or shape of input (list)
        # preload_data                n dict/None - training data to be preloaded to TensorFlow constant
        # n_s                         n int       - total number of simulations
        # fiducial_θ                  n array     - fiducial parameter values to be loaded as a TensorFlow constant
        # derivative_denominator      n array     - derivative denominator values to be loaded as TensorFlow constant
        # n_params                    n int       - number of parameters in the model
        # n_p                         n int       - number of differentiation simulations
        # prebuild                    n bool      - True to allow IMNN to build the network
        # verbose                     n bool      - True to print outputs such as shape of tensors
        # get_MLE                     n bool      - True to calculate MLE
        #______________________________________________________________
        # VARIABLES
        # x                           n tensor    - fiducial simulation input tensor
        # central_indices             n tensor    - list of indices to select preloaded central data at
        # derivative_indices          n tensor    - list of indices to select preloaded derivative data at
        # x_central                   n tensor    - input tensor for fiducial simulations
        # central_input                 tensor    - fiducial simulations to feed through network to calculate fisher
        # x_m                         n tensor    - below fiducial simulation input tensor
        # derivative_input_m            tensor    - lower simulations to feed through network to calculate fisher
        # x_p                         n tensor    - above fiducial simulation input tensor
        # derivative_input_m            tensor    - above simulations to feed through network to calculate fisher
        # test_input                    tensor    - test simulations to feed through network to calculate fisher
        # test_derivative_input_p       tensor    - lower test simulations to feed through network to calculate fisher
        # test_derivative_input_p       tensor    - above test simulations to feed through network to calculate fisher
        # θ_fid                       n tensor    - fiducial parameter input tensor for calculating MLE
        # prior                       n tensor    - prior range for each parameter for calculating MLE
        # dd                          n tensor    - inverse difference between upper and lower parameter value
        # dropout                     n tensor    - keep rate for dropout layer
        # output                        tensor    - network output for simulations at fiducial parameter value
        # output                      n tensor    - network output for simulations at fiducial parameter value (named)
        # output_central                tensor    - network output for simulations at fiducial parameter value to calculate fisher
        # output_m                      tensor    - network output for simulations below fiducial parameter value to calculate fisher
        # output_p                      tensor    - network output for simulations above fiducial parameter value to calculate fisher
        # test_output_central           tensor    - network output for test simulations at fiducial parameter value to calculate fisher
        # test_output_m                 tensor    - network output for test simulations below fiducial parameter value to calculate fisher
        # test_output_p                 tensor    - network output for test simulations above fiducial parameter value to calculate fisher
        # F                           n tensor    - Fisher information matrix
        # iC                          n tensor    - inverse covariance of network outputs for fiducial simulations
        # μ                           n tensor    - mean of network outputs for fiducial simulations
        # dμdθ                        n tensor    - numerical derivative of the mean of network outputs
        # C                           n tensor    - covariance of network outputs for fiducial test simulations
        # Λ                           n tensor    - loss function
        # test_F                      n tensor    - Fisher information matrix from test simulations
        # test_iC                     n tensor    - inverse covariance of network outputs for fiducial test simulations
        # test_μ                      n tensor    - mean of network outputs for fiducial test simulations
        # test_dμdθ                   n tensor    - numerical derivative of the mean of network outputs from test simulations
        # test_C                      n tensor    - covariance of network outputs for fiducial test simulations
        # test_Λ                      n tensor    - loss function from test simulations
        # MLE                         n tensor    - MLE of parameters
        # AL                          n tensor    - asymptotic likelihood at range of parameter values
        #______________________________________________________________
        n.x = tf.placeholder(n._FLOATX, shape = [None] + n.inputs, name = "x")
        if n.preload_data is not None:
            n.central_indices = tf.placeholder(tf.int32, shape = [n.n_s, 1], name = "central_indices")
            n.derivative_indices = tf.placeholder(tf.int32, shape = [n.n_p, 1], name = "derivative_indices")
            n.x_central = tf.constant(n.preload_data["x_central"], dtype = n._FLOATX)
            central_input = tf.gather_nd(n.x_central, n.central_indices)
            n.x_m = tf.constant(n.preload_data["x_m"], dtype = n._FLOATX)
            n.x_m = tf.stop_gradient(n.x_m)
            derivative_input_m = tf.reshape(tf.gather_nd(n.x_m, n.derivative_indices), [n.n_p * n.n_params] + n.inputs)
            n.x_p = tf.constant(n.preload_data["x_p"], dtype = n._FLOATX)
            n.x_p = tf.stop_gradient(n.x_p)
            derivative_input_p = tf.reshape(tf.gather_nd(n.x_p, n.derivative_indices), [n.n_p * n.n_params] + n.inputs)
            if set(["x_central_test", "x_m_test", "x_p_test"]).issubset(n.preload_data.keys()):
                test_input = tf.constant(n.preload_data["x_central_test"], dtype = n._FLOATX)
                test_derivative_input_m = tf.reshape(tf.constant(n.preload_data["x_m_test"], dtype = n._FLOATX), [n.n_p * n.n_params] + n.inputs)
                test_derivative_input_p = tf.reshape(tf.constant(n.preload_data["x_p_test"], dtype = n._FLOATX), [n.n_p * n.n_params] + n.inputs)
            else:
                test_input = None
        else:
            n.x_central = tf.placeholder(n._FLOATX, shape = [n.n_s] + n.inputs, name = "x_central")
            central_input = tf.identity(n.x_central)
            n.x_m = tf.placeholder(n._FLOATX, shape = [n.n_p, n.n_params] + n.inputs, name = "x_m")
            n.x_m = tf.stop_gradient(n.x_m)
            derivative_input_m = tf.reshape(n.x_m, [n.n_p * n.n_params] + n.inputs)
            n.x_p = tf.placeholder(n._FLOATX, shape = [n.n_p, n.n_params] + n.inputs, name = "x_p")
            n.x_p = tf.stop_gradient(n.x_p)
            derivative_input_p = tf.reshape(n.x_p, [n.n_p * n.n_params] + n.inputs)
        if n.get_MLE:
            n.prior = tf.placeholder(dtype = n._FLOATX, shape = [None, n.n_params, 1000], name = "prior")
        n.θ_fid = tf.constant(n.fiducial_θ, dtype = n._FLOATX, name = "fiducial")
        n.dd = tf.constant(n.derivative_denominator, dtype = n._FLOATX, name = "dd")
        n.dropout = tf.placeholder(n._FLOATX, shape = (), name = "dropout")
        if n.prebuild:
            network = n.build_network
        utils.utils().to_prebuild(network)
        with tf.variable_scope("IMNN") as scope:
            output = network(n.x, n.dropout)
        n.output = tf.identity(output, name = "output")
        if n.verbose: print(n.output)
        with tf.variable_scope("IMNN") as scope:
            scope.reuse_variables()
            output_central = network(central_input, n.dropout)
            scope.reuse_variables()
            output_m = network(derivative_input_m, n.dropout)
            scope.reuse_variables()
            output_p = network(derivative_input_p, n.dropout)
            if n.preload_data is not None and test_input is not None:
                scope.reuse_variables()
                test_output_central = network(test_input, 1.)
                scope.reuse_variables()
                test_output_m = network(test_derivative_input_m, 1.)
                scope.reuse_variables()
                test_output_p = network(test_derivative_input_p, 1.)
            else:
                test_output_central = output_central
                test_output_m = output_m
                test_output_p = output_p
        F, iC, μ, dμdθ, C = n.Fisher(output_central, output_m, output_p)
        n.F = tf.identity(F, name = 'fisher_information')
        if n.verbose: print(n.F)
        n.iC = tf.identity(iC, name = 'inverse_covariance')
        n.C = tf.identity(C, name = 'covariance')
        n.μ = tf.identity(μ, name = 'mean')
        n.dμdθ = tf.identity(dμdθ, name = 'mean_derivative')
        n.Λ = tf.identity(n.loss(n.F), name = 'loss')
        test_F, test_iC, test_μ, test_dμdθ, test_C  = n.Fisher(test_output_central, test_output_m, test_output_p)
        n.test_F = tf.identity(test_F, name = "test_F")
        n.test_iC = tf.identity(test_iC, name = 'test_inverse_covariance')
        n.test_C = tf.identity(test_C, name = 'test_covariance')
        n.test_μ = tf.identity(test_μ, name = 'test_mean')
        n.test_dμdθ = tf.identity(test_dμdθ, name = 'test_mean_derivative')
        n.test_Λ = tf.identity(n.loss(n.test_F), name = 'test_loss')
        if n.get_MLE:
            n.MLE = tf.identity(n.maximum_likelihood_estimate(), name = "maximum_likelihood_estimate")
            if n.verbose: print(n.MLE)
            n.AL = tf.identity(n.calculate_asymptotic_likelihood(), name = "asymptotic_likelihood")
            if n.verbose: print(n.AL)
        n.training_scheme(η)
        n.begin_session()

    def training_scheme(n, η):
        # MINIMISATION SCHEME FOR BACKPROPAGATION
        #______________________________________________________________
        # FUNCTIONS (DEFINED IN IMNN.py)
        # loss(tensor)                  tensor    - calculates loss function
        #______________________________________________________________
        # FUNCTIONS (DEFINED IN utils.py)
        # isfloat(list)                 float     - checks that parameter is a float
        #______________________________________________________________
        # INPUTS
        # η                             float     - learning rate
        # F                           n tensor    - Fisher information matrix
        #______________________________________________________________
        # VARIABLES
        # Λ                           n tensor    - loss function to minimise
        # backpropagate               n tf opt    - minimisation scheme for the network
        #______________________________________________________________
        η = utils.utils().isfloat(η, key = 'η')
        n.backpropagate = tf.train.AdamOptimizer(η, epsilon = 1.).minimize(n.Λ)
        #n.backpropagate = tf.train.GradientDescentOptimizer(η).minimize(n.loss(n.F))

    def train(n, num_epochs, n_train, keep_rate, history = True, data = None):
        # TRAIN INFORMATION MAXIMISING NEURAL NETWORK
        #______________________________________________________________
        # RETURNS
        # list or list, list
        # determinant of Fisher information matrix at the end of each epoch for train data (and test data)
        #______________________________________________________________
        # FUNCTIONS (DEFINED IN IMNN.py)
        # save_network(optional Bool)             - saves the network (if n.save_file is not None)
        #______________________________________________________________
        # FUNCTIONS (DEFINED IN utils.py)
        # positive_integer(list)        int       - checks that parameter is a positive integer
        # constrained_float(float, string)
        #                               float     - checks the dropout is between 0 and 1
        #______________________________________________________________
        # INPUTS
        # num_epochs                    int       - number of epochs to train
        # n_train                       int       - number of combinations to split training set into
        # keep_rate                     float     - keep rate for dropout
        # history                       bool      - whether to collect training history
        # data                          dict      - data at fiducial, lower and upper parameter values (with test versions)
        # sess                        n session   - interactive tensorflow session (with initialised parameters)
        # backpropagate               n tf opt    - minimisation scheme for the network
        # n_s                         n int       - total number of simulations
        # n_p                         n int       - number of differentiation simulations
        # x_central                   n tensor    - fiducial simulation input tensor
        # central_indices             n tensor    - list of indices to select preloaded central data at
        # derivative_indices          n tensor    - list of indices to select preloaded derivative data at
        # x_m                         n tensor    - below fiducial simulation input tensor
        # x_p                         n tensor    - above fiducial simulation input tensor
        # dropout                     n tensor    - keep rate for dropout layer
        # F                           n tensor    - Fisher information matrix
        # test_F                      n tensor    - Fisher information matrix from test simulations
        # save_file                   n str/None  - Name to save or load graph. None does not save graph
        #______________________________________________________________
        # VARIABLES
        # test                          bool      - True if using test_data
        # history                     n history   - dictionary of history of various parameter during training
        # tq                            tqdm loop - a looper in tqdm for showing the progress bar
        # epoch                         int       - epoch counter
        # central_indices               array     - list of indices at which to select the central data for training
        # derivative_indices            array     - list of indices at which to select the derivative data for training
        # combination                   int       - counter for number of individual combinations of simulations
        #______________________________________________________________
        num_epochs = utils.utils().positive_integer(num_epochs, key = 'number of epochs')
        n_train = utils.utils().positive_integer(n_train, key = 'number of combinations')
        keep_rate = utils.utils().constrained_float(keep_rate, key = 'dropout')

        n.history = {}
        n.history["F"] = []
        n.history["det(F)"] = []
        if history:
            n.history["Λ"] = []
            n.history["μ"] = []
            n.history["C"] = []
            n.history["det(C)"] = []
            n.history["dμdθ"] = []
            if n.x_central.op.type != 'Placeholder':
                data = n.preload_data
            if 'x_central_test' in data.keys():
                test = True
                n.history["test F"] = []
                n.history["det(test F)"] = []
                n.history["test Λ"] = []
                n.history["test μ"] = []
                n.history["test C"] = []
                n.history["det(test C)"] = []
                n.history["test dμdθ"] = []
            else:
                test = False

        central_indices = np.arange(n.n_s * n_train)
        derivative_indices = np.arange(n.n_p * n_train)
        tq = tqdm.trange(num_epochs)
        for epoch in tq:
            np.random.shuffle(central_indices)
            np.random.shuffle(derivative_indices)
            if n.x_central.op.type != 'Placeholder':
                for combination in range(n_train):
                    n.sess.run(n.backpropagate, feed_dict = {n.central_indices: central_indices[combination * n.n_s: (combination + 1) * n.n_s].reshape((n.n_s, 1)), n.derivative_indices: derivative_indices[combination * n.n_p: (combination + 1) * n.n_p].reshape((n.n_p, 1)), n.dropout: keep_rate})
                train_F = n.sess.run(n.F, feed_dict = {n.central_indices: central_indices[combination * n.n_s: (combination + 1) * n.n_s].reshape((n.n_s, 1)), n.derivative_indices: derivative_indices[combination * n.n_p: (combination + 1) * n.n_p].reshape((n.n_p, 1)), n.dropout: 1.})
                det_train_F = np.linalg.det(train_F)
            else:
                for combination in range(n_train):
                    n.sess.run(n.backpropagate, feed_dict = {n.x_central: data['x_central'][central_indices[combination * n.n_s: (combination + 1) * n.n_s]], n.x_m: data['x_m'][derivative_indices[combination * n.n_p: (combination + 1) * n.n_p]], n.x_p: data['x_p'][derivative_indices[combination * n.n_p: (combination + 1) * n.n_p]], n.dropout: keep_rate})
                train_F = n.sess.run(n.F, feed_dict = {n.x_central: data['x_central'][central_indices[combination * n.n_s: (combination + 1) * n.n_s]], n.x_m: data['x_m'][derivative_indices[combination * n.n_p: (combination + 1) * n.n_p]], n.x_p: data['x_p'][derivative_indices[combination * n.n_p: (combination + 1) * n.n_p]], n.dropout: 1.})
                det_train_F = np.linalg.det(train_F)
            n.history["F"].append(train_F)
            n.history["det(F)"].append(det_train_F)
            if history:
                if test:
                    if n.x_central.op.type != 'Placeholder':
                        μ, C, dμdθ, Λ, test_F, test_μ, test_C, test_dμdθ, test_Λ = n.sess.run([n.μ, n.C, n.dμdθ, n.Λ, n.test_F, n.test_μ, n.test_C, n.test_dμdθ, n.test_Λ], feed_dict = {n.central_indices: central_indices[combination * n.n_s: (combination + 1) * n.n_s].reshape((n.n_s, 1)), n.derivative_indices: derivative_indices[combination * n.n_p: (combination + 1) * n.n_p].reshape((n.n_p, 1)), n.dropout: 1.})
                    else:
                        μ, C, dμdθ, Λ, test_F, test_μ, test_C, test_dμdθ, test_Λ = n.sess.run([n.μ, n.C, n.dμdθ, n.Λ, n.test_F, n.test_μ, n.test_C, n.test_dμdθ, n.test_Λ], feed_dict = {n.x_central: data['x_central_test'], n.x_m: data['x_m_test'], n.x_p: data['x_p_test'], n.dropout: 1.})
                    n.history["test F"].append(test_F)
                    n.history["det(test F)"].append(np.linalg.det(test_F))
                    n.history["test μ"].append(test_μ)
                    n.history["test C"].append(test_C)
                    n.history["det(test C)"].append(np.linalg.det(test_C))
                    n.history["test dμdθ"].append(test_dμdθ)
                    n.history["test Λ"].append(test_Λ)
                    tq.set_postfix(detF = n.history["det(F)"][-1], detF_test = n.history["det(test F)"][-1])
                else:
                    if n.x_central.op.type != 'Placeholder':
                        μ, C, dμdθ, Λ = n.sess.run([n.μ, n.C, n.dμdθ, n.Λ], feed_dict = {n.central_indices: central_indices[combination * n.n_s: (combination + 1) * n.n_s].reshape((n.n_s, 1)), n.derivative_indices: derivative_indices[combination * n.n_p: (combination + 1) * n.n_p].reshape((n.n_p, 1)), n.dropout: 1.})
                    else:
                        μ, C, dμdθ, Λ = n.sess.run([n.μ, n.C, n.dμdθ, n.Λ], feed_dict = {n.x_central: data['x_central'][central_indices[combination * n.n_s: (combination + 1) * n.n_s]], n.x_m: data['x_m'][derivative_indices[combination * n.n_p: (combination + 1) * n.n_p]], n.x_p: data['x_p'][derivative_indices[combination * n.n_p: (combination + 1) * n.n_p]], n.dropout: 1.})
                    tq.set_postfix(detF = n.history["det(F)"][-1])
                n.history["μ"].append(μ)
                n.history["C"].append(C)
                n.history["det(C)"].append(np.linalg.det(C))
                n.history["dμdθ"].append(dμdθ)
                n.history["Λ"].append(Λ)
            else:
                tq.set_postfix(detF = n.history["det(F)"][-1])
        if n.save_file is not None:
            n.save_network()

    def ABC(n, real_data, prior, draws, generate_simulation, at_once = True, data = None):
        # PERFORM APPROXIMATE BAYESIAN COMPUTATION WITH RANDOM DRAWS FROM PRIOR
        #______________________________________________________________
        # CALLED FROM
        # PMC(array, list, int, int, func, float, optional bool, optional list, optional array)
        #                               array, float, array, array, array, int, array
        #                                         - performs population monte carlo
        #______________________________________________________________
        # RETURNS
        # array, float, array, array, array
        # sampled parameter values, network summary of real data, summaries of simulations,
        #     distances between simulation summaries, real summary and Fisher information
        #______________________________________________________________
        # INPUTS
        # real_data                     array     - real data to be summarised
        # prior                         list      - lower and upper bound of uniform prior
        # draws                         int       - number of draws from the prior
        # generate_simulation           func      - function which generates the simulation at parameter value
        # at_once              optional bool      - True if generate all simulations and calculate all summaries at once
        # data                 optional dict      - dictionary containing test data
        # sess                        n session   - interactive tensorflow session (with initialised parameters)
        # test_F                      n tensor    - Fisher information matrix
        # x                           n tensor    - fiducial simulation input tensor
        # test_F                      n tensor    - Fisher information from test data
        # x_central                   n tensor    - input tensor for fiducial simulations
        # x_m                         n tensor    - below fiducial simulation input tensor
        # x_p                         n tensor    - above fiducial simulation input tensor
        # dropout                     n tensor    - keep rate for dropout layer
        # output                      n tensor    - network output for simulations at fiducial parameter value
        # n_summaries                 n int       - number of outputs from the network
        #______________________________________________________________
        # VARIABLES
        # F                             array     - Fisher information matrix from test simulations
        # summary                       float     - summary of the real data
        # θ                             array     - all draws from prior
        # simulations                   array     - generated simulations at all parameter draws
        # simulation_summaries          float     - summaries of each of the generated simulations
        # theta                         int       - counter for number of draws
        # simulation                    array     - generated simulation at indivdual parameter draw
        # difference                    array     - difference between summary of real data and summaries of simulations
        # distances                     array     - Euclidean distance between real summary and summaries of simulations
        #______________________________________________________________
        if n.x_central.op.type != 'Placeholder':
            F = n.sess.run(n.test_F)
        else:
            F = n.sess.run(n.test_F, feed_dict = {n.x_central: data['x_central_test'], n.x_m: data['x_m_test'], n.x_p: data['x_p_test'], n.dropout: 1.})
        summary = n.sess.run(n.output, feed_dict = {n.x: real_data, n.dropout: 1.})[0]
        θ = np.random.uniform(prior[0], prior[1], draws)
        if at_once:
            simulations = generate_simulation(θ)
            simulation_summaries = n.sess.run(n.output, feed_dict = {n.x: simulations, n.dropout: 1.})
        else:
            simulation_summaries = np.zeros([draws, n.n_summaries])
            for theta in tqdm.tqdm(range(draws)):
                simulation = generate_simulation([θ[theta]])
                simulation_summaries[theta] = n.sess.run(n.output, feed_dict = {n.x: simulation, n.dropout: 1.})[0]
        difference = simulation_summaries - summary
        distances = np.sqrt(np.einsum('ij,ij->i', difference, np.einsum('jk,ik->ij', F, difference)))
        return θ, summary, simulation_summaries, distances, F

    def PMC(n, real_data, prior, num_draws, num_keep, generate_simulation, criterion, at_once = True, samples = None, data = None):
        # PERFORM APPROXIMATE BAYESIAN COMPUTATION USING POPULATION MONTE CARLO
        #______________________________________________________________
        # RETURNS
        # array, float, array, array, array, int, array
        # sampled parameter values, network summary of real data, distances between simulation summaries and real summary,
        #    summaries of simulations, weighting of samples, total number of draws so far and Fisher information
        #______________________________________________________________
        # FUNCTIONS (DEFINED IN utils.py)
        # to_continue(list)             bool      - True if continue running PMC
        # enough(float/int, float/int, optional bool, optional bool)
        #                                         - checks that first value is higher than second
        #______________________________________________________________
        # INPUTS
        # real_data                     array     - real data to be summarised
        # prior                         list      - lower and upper bound of uniform prior
        # num_draws                     int       - number of initial draws from the prior
        # num_keep                      int       - number of samples in the approximate posterior
        # generate_simulation           func      - function which generates the simulation at parameter value
        # criterion                     float     - ratio of number of draws wanted over number of draws needed
        # at_once              optional bool      - True if generate all simulations and calculate all summaries at once
        # samples              optional list      - list of all the outputs of PMC to continue running PMC
        # data                 optional dict      - dictionary containing test data
        # sess                        n session   - interactive tensorflow session (with initialised parameters)
        # output                      n tensor    - network output for simulations at fiducial parameter value
        # x                           n tensor    - fiducial simulation input tensor
        # dropout                     n tensor    - keep rate for dropout layer
        # n_summaries                 n int       - number of outputs from the network
        #______________________________________________________________
        # VARIABLES
        # continue_from_samples         bool      - True is continue running the PMC
        # θ_                            array     - parameter draws from approximate posterior
        # summary                       float     - summary of the real data
        # ρ_                            array     - distance between real summary and summaries from approximate posterior
        # s_                            array     - summaries from approximate posterior
        # W                             array     - weighting for samples in approximate posterior
        # total_draws                   int       - total number of draws from the prior so far
        # F                             array     - Fisher information matrix of test simulations
        # θ                             array     - all parameter draws from prior
        # s                             float     - summaries of each of the generated simulations from prior
        # ρ                             array     - distance between real summary and summaries of simulations from prior
        # keep_index                    array     - sorted indices of closest compressed simulations to real data
        # pθ                            float     - value of prior at given θ value (for uniform prior)
        # iteration                     int       - counter for number of iterations of PMC
        # criterion_reached             float     - number of draws wanted over number of current draws
        # cov                           array     - weighted covariance of drawn samples
        # ϵ                             float     - current accept distance
        # redraw_index                  array     - indices of parameter draws to redraw
        # W_temp                        array     - copy of sample weighting array for updating
        # current_draws                 int       - number of parameters being updated at once
        # draws                         int       - counter for number of draws per iteration
        # θ_temp                        array     - proposed updated parameter sample values
        # below_prior                   array     - indices of parameter values below prior
        # above_prior                   array     - indices of parameter values above prior
        # simulations                   array     - generated simulations at all parameter draws
        # simulation_summaries          float     - summaries of each of the generated simulations
        # theta                         int       - counter for number of draws
        # simulation                    array     - generated simulation at indivdual parameter draw
        # difference                    array     - difference between summary of real data and summaries of simulations
        # ρ_temp                        array     - distance between real and simulated summaries for proposed draws
        # accept_index                  array     - indices of proposed parameter values to keep
        #______________________________________________________________
        continue_from_samples = utils.utils().to_continue(samples)
        utils.utils().enough(num_draws, num_keep)
        if continue_from_samples:
            θ_ = samples[0]
            summary = samples[1]
            ρ_ = samples[2]
            s_ = samples[3]
            W = samples[4]
            total_draws = samples[5]
            F = samples[6]
        else:
            θ, summary, s, ρ, F = n.ABC(real_data, prior, num_draws, generate_simulation, at_once, data = data)
            keep_index = np.argsort(np.abs(ρ))
            θ_ = θ[keep_index[: num_keep]]
            ρ_ = ρ[keep_index[: num_keep]]
            s_ = s[keep_index[: num_keep]]
            W = np.ones(num_keep) / num_keep
            total_draws = num_draws
        pθ = 1./(prior[1] - prior[0])
        iteration = 0
        criterion_reached = 1e10
        while criterion < criterion_reached:
            cov = np.cov(θ_, aweights = W)
            ϵ = np.percentile(ρ_, 75)
            redraw_index = np.where(ρ_ >= ϵ)[0]
            W_temp = np.copy(W)
            current_draws = len(redraw_index)
            draws = 0
            while current_draws > 0:
                draws += current_draws
                θ_temp = np.random.normal(θ_[redraw_index], np.sqrt(cov))
                below_prior = np.where(θ_temp <= prior[0])[0]
                above_prior = np.where(θ_temp > prior[1])[0]
                while len(below_prior) > 0 or len(above_prior) > 0:
                    θ_temp[below_prior] = np.random.normal(θ_[redraw_index[below_prior]], np.sqrt(cov))
                    θ_temp[above_prior] = np.random.normal(θ_[redraw_index[above_prior]], np.sqrt(cov))
                    below_prior = np.where(θ_temp <= prior[0])[0]
                    above_prior = np.where(θ_temp > prior[1])[0]
                if at_once:
                    simulations = generate_simulation(θ_temp)
                    simulation_summaries = n.sess.run(n.output, feed_dict = {n.x: simulations, n.dropout: 1.})
                else:
                    simulation_summaries = np.zeros([current_draws, n.n_summaries])
                    for theta in range(current_draws):
                        simulation = generate_simulation([θ_temp[theta]])
                        simulation_summaries[theta] = n.sess.run(n.output, feed_dict = {n.x: simulation, n.dropout: 1.})[0]
                difference = simulation_summaries - summary
                ρ_temp = np.sqrt(np.einsum('ij,ij->i', difference, np.einsum('jk,ik->ij', F, difference)))
                accept_index = np.where(ρ_temp <= ϵ)[0]
                if len(accept_index) > 0:
                    ρ_[redraw_index[accept_index]] = ρ_temp[accept_index]
                    θ_[redraw_index[accept_index]] = θ_temp[accept_index]
                    s_[redraw_index[accept_index]] = simulation_summaries[accept_index]
                    W_temp[redraw_index[accept_index]] = pθ / np.sum(W[:, None] * np.exp(-0.5 * (np.stack([θ_temp[accept_index] for i in range(num_keep)]) - θ_[:, None])**2. / cov) / np.sqrt(2 * np.pi * cov), axis = 0)
                redraw_index = np.where(ρ_ >= ϵ)[0]
                current_draws = len(redraw_index)
            W = np.copy(W_temp)
            criterion_reached = num_keep / draws
            iteration += 1
            total_draws += draws
            print('iteration = ' + str(iteration) + ', current criterion = ' + str(criterion_reached) + ', total draws = ' + str(total_draws) + ', ϵ = ' + str(ϵ) + '.', end = '\r')
        return θ_, summary, ρ_, s_, W, total_draws, F

    def θ_MLE(n, real_data, data = None):
        # CALCULATE MAXIMUM LIKELIHOOD ESTIMATE OF PARAMETERS
        #______________________________________________________________
        # RETURNS
        # array
        # Maximum likelihood estimate of parameters
        #______________________________________________________________
        # INPUTS
        # real_data                     array     - real data to be summarised
        # data                 optional dict      - dictionary containing test data
        # sess                        n session   - interactive tensorflow session (with initialised parameters)
        # x                           n tensor    - fiducial simulation input tensor
        # x_central                   n tensor    - input tensor for fiducial simulations
        # x_m                         n tensor    - below fiducial simulation input tensor
        # x_p                         n tensor    - above fiducial simulation input tensor
        # dropout                     n tensor    - keep rate for dropout layer
        # MLE                         n tensor    - MLE of parameters
        #______________________________________________________________
        if n.x_central.op.type != 'Placeholder':
            return n.sess.run(n.MLE, feed_dict = {n.x: real_data, n.dropout: 1.})
        else:
            return n.sess.run(n.MLE, feed_dict = {n.x: real_data, n.x_central: data['x_central_test'], n.x_m: data['x_m_test'], n.x_p: data['x_p_test'], n.dropout: 1.})

    def asymptotic_likelihood(n, real_data, prior, data = None):
        # CALCULATE ASYMPTOTIC LIKELIHOOD OVER PARAMETER RANGE
        #______________________________________________________________
        # RETURNS
        # array
        # Asymptotic likelihood over a parameter range
        #______________________________________________________________
        # INPUTS
        # real_data                     array     - real data to be summarised
        # prior                         array     - range of parameter values to calculate asymptotic likelihood at
        # data                 optional dict      - dictionary containing test data
        # sess                        n session   - interactive tensorflow session (with initialised parameters)
        # x                           n tensor    - fiducial simulation input tensor
        # prior                       n tensor    - prior range for each parameter for calculating MLE
        # x_central                   n tensor    - input tensor for fiducial simulations
        # x_m                         n tensor    - below fiducial simulation input tensor
        # x_p                         n tensor    - above fiducial simulation input tensor
        # dropout                     n tensor    - keep rate for dropout layer
        # AL                          n tensor    - asymptotic likelihood at range of parameter values
        #______________________________________________________________
        if n.x_central.op.type != 'Placeholder':
            return n.sess.run(n.AL, feed_dict = {n.x: real_data, n.prior: prior, n.dropout: 1.})
        else:
            return n.sess.run(n.AL, feed_dict = {n.x: real_data, n.x_central: data['x_central_test'], n.x_m: data['x_m_test'], n.x_p: data['x_p_test'], n.prior: prior, n.dropout: 1.})
