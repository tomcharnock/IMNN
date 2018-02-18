import tensorflow as tf
import numpy as np
import sys
import tqdm
import operator
import scipy.optimize as so

class network(): 
    def __init__(n, parameters):
        # INITIALISE NETWORK PARAMETERS
        # parameters            dict       - dictionary containing initialisation parameters
        #______________________________________________________________
        # u                   n class      - utility functions
        # _FLOATX             n TF type    - set TensorFlow types to 32 bit (for GPU)
        # tot_sims            n int        - total number of simulations to use
        # _                     bool       - empty error holder
        # n_params            n int        - number of outputs from the network
        # inputs              n int        - number of inputs to the network
        # n_train             n int        - number of combinations to split training set into
        # n_s                 n int        - number of simulations in each combination
        # n_p                 n int        - number of differentiation simulations in each combination
        # n_batches           n int        - number of batches to calculate at one time
        # alpha_              n float      - value of the ReLU running parameter
        # b_bias              n float      - value to initialise the bias with
        # lr_                 n float      - value of the learning rate 
        # d                   n float      - value of the dropout fraction
        # params              n bool       - switch to choose which direction to calculate the covariance
        # act_dict              dict       - holds functions for activation function selection
        # activation          n function   - activation function to use
        # l                   n list       - contains the neural architecture of the network
        # der_den_            n [n_params] - denominator values for numerical derivative
        # n.n_sp              n int (conv) - index of simulations + lower derivatives
        # n.n_spp             n int (conv) - index of simulations = all derivatives
        # n.n_pp              n int (conv) - index of all derivatives
        # n.partial_sims      n int (conv) - index of partial derivatives 
        #______________________________________________________________
        n.u = utils()
        n._FLOATX = tf.float32
        n.tot_sims = parameters['total number of simulations']
        _ = n.u.check_error(n.tot_sims, [int, operator.le, 0], 'Total number of simulations must be a positive integer')
        n.n_params = parameters['number of parameters']
        _ = n.u.check_error(n.n_params, [int, operator.le, 0], 'Number of parameters must be a positive integer')
        _ = n.u.check_error(n.n_params, [operator.gt, 1], 'More than 1 parameter is allowed, but may not be properly implemented', to_break = False)
        n.inputs = parameters['number of inputs']
        _ = n.u.check_error(n.inputs, [int, operator.le, 0], 'Number of inputs must be a positive integer')
        n.n_train = parameters['number of combinations']
        _ = n.u.check_error(n.n_train, [int, operator.le, 0, operator.gt, n.tot_sims], 'Number of combinations must be a positive integer and less than the total number of simulations')
        n.n_s = n.tot_sims / n.n_train
        _ = n.u.check_error(float(int(n.n_s)), [operator.ne, n.n_s], 'Total number of simulations cannot be used')
        n.n_s = int(n.n_s)
        if _:
            print('Now using ', n.n_s * n.n_train, ' out of ', n.tot_sims)
        n.n_p = int(n.n_s * parameters['differentiation fraction'])
        _ = n.u.check_error(n.n_p, ['compare', operator.le, 0, operator.gt, n.n_s], 'The fraction of differentiation simulations which can be used must be between 0 and 1')
        n.n_batches = parameters['number of batches']
        _ = n.u.check_error(n.n_batches, [int, operator.le, 0, operator.gt, n.n_train], 'Number of batches must be a positive integer less than the possible number of combinations')
        n.alpha_ = parameters['alpha']
        _ = n.u.check_error(n.alpha_, [float], 'The value of the relu running parameter should be a float')
        n.b_bias = parameters['biases bias']
        _ = n.u.check_error(n.b_bias, [float], 'The initial bias of each of the biases should be a float')
        n.lr_ = parameters['learning rate']
        _ = n.u.check_error(n.lr_, [float], 'The learning rate of should be a small float')
        n.d = parameters['dropout']
        _ = n.u.check_error(n.d, [float, 'compare', operator.lt, 0., operator.ge, 1.], 'Dropout should be float between 0 and 1')
        n.params = parameters['parameter direction']
        _ = n.u.check_error(n.params, [bool], 'The direction for the multi-parameter covariance calculation should be a boolean. This parameter is not important when only considering one parameter')
        act_dict = {'relu': n.relu, 'sigmoid': n.sigmoid, 'tanh': n.tanh, 'softplus': n.softplus}
        if (parameters['activation function'] not in act_dict):
            print('The activation function must be one of relu, sigmoid, tanh, or softplus')
            sys.exit()
        else:
            n.activation = act_dict[parameters['activation function']]
        if (type(parameters['hidden layers']) != list):
            print('The architecture must be a list of layers with the number of nodes in each layer')
            sys.exit()
        if (len(parameters['hidden layers']) <= 0):
            print('The number of hidden layers must be positive')
            sys.exit()
        if any(type(i) != int for i in parameters['hidden layers']):
            print('The number of neurons in each layer must be a positive integer')
            sys.exit()
        if any(i <= 0 for i in parameters['hidden layers']):
            print('The number of neurons in each layer must be a positive integer')
            sys.exit()
        n.l = [n.inputs] + parameters['hidden layers'] + [n.n_params]    
        n.der_den_ = parameters['denominator for the derivative']
        n.n_sp = n.n_s + n.n_params * n.n_p
        n.n_spp = n.n_sp + n.n_params * n.n_p
        n.n_pp = n.n_p * n.n_params
        n.partial_sims = n.n_train * n.n_p
 
    def setup(n):
        # SETUP NEURAL NETWORK ARCHITECTURE
        #______________________________________________________________
        # feedforward(x, W, b, dropped)   (a, dadv)            - feedforward inputs to get outputs and derivatives
        # fisher(a[-1])                   (|F|, cov)           - calculate determinant of Fisher and covariance from last layer
        # central_values(a, dadv)         (a_, dadv_)          - unpack simulations to get only the non-derivative-sim outputs
        # backpropagate_and_update(W, b, a_, dadv_, |F|, cov, dropped)
        #                                 (weights_and_biases) - update weights and biases using Fisher information as loss
        # session(training_graph)         (training_session)   - initialise TensorFlow session for training
        #______________________________________________________________
        # training_graph     n TF obj                              - TensorFlow graph for training the network
        # W                    list (TF variable)                  - neural network weights
        # b                    list (TF variable)                  - neural network biases
        # dropped              list (TF placeholder)               - holder for which neurons to drop
        # x                    list (TF placeholder)               - holder for network input
        # l                    int                                 - layer counting variable
        # a                    list (TF tensor)                    - activated outputs at every neuron
        # dadv                 list (TF tensor)                    - derivative of activated output at every neuron
        # IFI                n TF variable [:]                     - determinant of the Fisher information
        # cov                  TF variable [:, n_params, n_params] - covariance of network outputs
        # a_                   list (TF tensor)                    - unpacked outputs with only non-derivative simulations
        # dadv_                list (TF tensor)                    - unpacked derivatives with only non-derivative simulations
        # weights_and_biases n list [|F|, W, b]                    - list with training outputs
        # train_session      n TF session                          - initialised TensorFlow session
        #______________________________________________________________
        n.training_graph = tf.Graph()
        with n.training_graph.as_default() as g:
            W = []
            b = []
            dropped = []
            x = tf.placeholder(n._FLOATX, shape = [None, n.n_spp, n.l[0]], name = 'x')
            for l in range(len(n.l) - 1):
                W.append(tf.Variable(tf.truncated_normal([n.l[l], n.l[l + 1]], mean = 0., stddev = tf.divide(tf.sqrt(2.), n.l[l]), dtype = n._FLOATX)))
                b.append(tf.Variable(tf.constant(n.b_bias, shape = [n.l[l + 1]], dtype = n._FLOATX)))
                dropped.append(tf.placeholder(n._FLOATX, shape = [n.l[l + 1]], name = 'd_' + str(l)))
            a, dadv = n.feedforward(x, W, b, dropped)
            n.IFI, cov = n.fisher(a[-1])
            a_, dadv_ = n.central_values(a, dadv)
            n.weights_and_biases = n.backpropagate_and_update(W, b, a_, dadv_, n.IFI, cov, dropped)
        n.train_session = n.session(n.training_graph)
        
    def feedforward(n, x, W, b, dropped):
        # FEEDFORWARD ALGORITHM
        # called from setup()
        # x                    list (TF placeholder)  - holder for network input
        # W                    list (TF variable)     - neural network weights
        # b                    list (TF variable)     - neural network biases
        # dropped              list (TF placeholder)  - holder for which neurons to drop
        # returns activated outputs and their derivatives (a, dadv)
        #______________________________________________________________
        # activation(v, l, dropped)  (a_, dadv_)      - activation function
        #______________________________________________________________
        # a                    list (TF tensor)       - activated outputs
        # dadv                 list (TF tensor)       - derivative of activated outputs
        # l                    int                    - layer counting variable
        # v                    TF variable            - weighted biased input
        # a_                   TF variable            - temporary activated outputs
        # dadv_                TF variable            - temporary activated derivative outputs
        #______________________________________________________________
        a = []
        dadv = []
        for l in range(len(n.l) - 1):
            if l == 0:
                a.append(x)
                dadv.append(x)
            v = tf.add(tf.einsum('ijk,kl->ijl', a[l], W[l]), b[l])
            a_, dadv_ = n.activation(v, l, dropped) #eval('n.' + n.activation + "(v, l, dropped)")
            a.append(a_)
            dadv.append(dadv_)
        return a, dadv 
    
    def sigmoid(n, v, l, dropped):
        # SIGMOID ACTIVATION FUNCTION
        # called from feedforward()
        # v                    TF variable            - weighted biased input
        # l                    int                    - layer counting variable
        # dropped              list (TF placeholder)  - holder for which neurons to drop
        # returns activated outputs and their derivatives where dropped neurons are zero (a, dadv)
        #______________________________________________________________
        # ones                 TF tensor              - TensorFlow ones
        # den                  TF variable            - denominator of activation (1 + exp(-v))
        # a_                   TF variable            - temporary activated output
        # sec                  TF variable            - term for calculating derivative (1 - a)
        # dadv                 TF variable            - derivative of output with dropout a(1 - a)
        # a                    TF variable            - activated output with dropout
        #______________________________________________________________
        ones = tf.ones([n.l[l + 1]], dtype = n._FLOATX)
        den = tf.add(ones, tf.exp(-v))
        a_ = tf.divide(ones, den)
        sec = tf.subtract(ones, a_)
        dadv = tf.einsum('ijk,k->ijk', tf.multiply(a_, sec), dropped[l]) 
        a = tf.einsum('ijk,k->ijk', a_, dropped[l])
        return a, dadv 

    def tanh(n, v, l, dropped):
        # TANH ACTIVATION FUNCTION
        # called from feedforward()
        # v                    TF variable            - weighted biased input
        # l                    int                    - layer counting variable
        # dropped              list (TF placeholder)  - holder for which neurons to drop
        # returns activated outputs and their derivatives where dropped neurons are zero (a, dadv)
        #______________________________________________________________
        # a_                   TF variable            - temporary activated output
        # dadv                 TF variable            - derivative of output with dropout (1 - a^2)
        # a                    TF variable            - activated output with dropout
        #______________________________________________________________
        a_ = tf.tanh(v)
        dadv = tf.einsum('ijk,k->ijk', tf.subtract(tf.ones([n.l[l + 1]], dtype = n._FLOATX), tf.square(a_)), dropped[l])
        a = tf.einsum('ijk,k->ijk', a_, dropped[l])
        return a, dadv
    
    def softplus(n, v, l, dropped):
        # SOFTPLUS ACTIVATION FUNCTION
        # called from feedforward()
        # v                    TF variable            - weighted biased input
        # l                    int                    - layer counting variable
        # dropped              list (TF placeholder)  - holder for which neurons to drop
        # returns activated outputs and their derivatives where dropped neurons are zero (a, dadv)
        #______________________________________________________________
        # ones                 TF tensor              - TensorFlow ones
        # a                    TF variable            - activated output with dropout ln(1 + exp(v))
        # den                  TF variable            - denominator of activation (1 + exp(-v))
        # dadv                 TF variable            - derivative of output with dropout 1 / (1 + exp(-v))
        #______________________________________________________________
        ones = tf.ones([n.l[l + 1]], dtype = n._FLOATX)
        a = tf.einsum('ijk,k->ijk', tf.log(tf.add(ones, tf.exp(v))), dropped[l])
        den = tf.add(ones, tf.exp(-v))
        dadv = tf.einsum('ijk,k->ijk', tf.divide(ones, den), dropped[l])
        return  a, dadv
    
    def relu(n, v, l, dropped):
        # RELU (LEAKY) ACTIVATION FUNCTION
        # v                    TF variable            - weighted biased input
        # l                    int                    - layer counting variable
        # dropped              list (TF placeholder)  - holder for which neurons to drop
        # called from feedforward()
        # returns activated outputs and their derivatives where dropped neurons are zero (a, dadv)
        #______________________________________________________________
        # ones                 TF tensor              - TensorFlow ones
        # alpha                TF constant            - value of the negative gradient
        # a                    TF variable            - activated output with dropout
        # dadv                 TF variable            - derivative of output with dropout
        #______________________________________________________________        
        ones = tf.ones_like(v)
        alpha = tf.constant(n.alpha_, shape = (), dtype = n._FLOATX)
        a = tf.einsum('ijk,k->ijk', tf.where(v > 0., x = v, y = tf.multiply(alpha, v)), dropped[l])
        dadv = tf.einsum('ijk,k->ijk', tf.where(v > 0., x = ones, y = tf.multiply(alpha, ones)), dropped[l])
        return  a, dadv
   
    def fisher(n, output):
        # CALCULATE THE FISHER INFORMATION
        # called from setup()
        # params              n bool                    - switch to choose which direction to calculate the covariance
        # output               TF variable              - the activated output at the last layer
        # returns determinant of the Fisher information matrix and the covariance of the network outputs
        #______________________________________________________________
        # unstack_data(output)      (a_, a_m, a_p)      - unstacks the network outputs into central, and plus and minus simulations
        # simulation_covariance(a_) (cov, mean)         - calculates the covariance of the central value simulation outputs
        # mean_derivative(a_m, a_p) (dmdt)              - calculates the numerical derivative of the mean of the simulations
        #______________________________________________________________
        # a_                  TF variable               - central simulation network outputs
        # a_m                 TF variable               - lower simulation network outputs
        # a_p                 TF variable               - upper simulation network outputs
        # cov                 TF variable               - covariance of central simulations
        # mean                TF variable               - mean of central simulations
        # invcov              TF variable               - inverse of the covariance matrix
        # dmdt                TF variable               - derivative of the mean of the network outputs
        # invcov_m_dmdt     n list [invcov, mean, dmdt] - list of Fisher components
        # F_non_sym           TF variable               - non-symmetric Fisher matrix
        # F                   TF variable               - Fisher information matrix
        # IFI                 TF variable               - determinant of the Fisher information
        #______________________________________________________________ 
        a_, a_m, a_p = n.unstack_data(output)
        cov, mean = n.simulation_covariance(a_)
        invcov = tf.matrix_inverse(cov)
        dmdt = n.mean_derivative(a_m, a_p)
        n.invcov_m_dmdt = [invcov, mean, dmdt]
        if n.params:
            F_non_sym = tf.multiply(tf.constant(0.5, shape = (), dtype = n._FLOATX), tf.einsum('ijk,ijl->ikl', dmdt, tf.einsum('ijk,ikl->ijl', invcov, dmdt)))
        else:
            F_non_sym = tf.multiply(tf.constant(0.5, shape = (), dtype = n._FLOATX), tf.einsum('ijk,ilk->ilj', dmdt, tf.einsum('ijk,ilk->ilj', invcov, dmdt)))
        F = tf.add(F_non_sym, tf.transpose(F_non_sym, perm = [0, 2, 1]))
        IFI = tf.matrix_determinant(F)
        return IFI, cov
    
    def unstack_data(n, data):
        # UNSTACK THE NETWORK OUTPUTS BY SIMULATION PARAMETER
        # called from fisher()
        # data                TF variable               - the activated output at the last layer
        # returns unstacked network outputs (a, a_m, a_p)
        #______________________________________________________________
        # a                   TF variable               - central simulation network outputs
        # a_m                 TF variable               - lower simulation network outputs
        # a_p                 TF variable               - upper simulation network outputs
        # param               int                       - parameter derivative counter
        #______________________________________________________________ 
        a = tf.slice(data, [0, 0, 0], [-1, n.n_s, -1])
        a_m_ = []
        a_p_ = []
        for param in range(n.n_params):
            a_m_.append(tf.slice(data, [0, n.n_s + param * n.n_p, 0], [-1, n.n_p, -1]))
            a_p_.append(tf.slice(data, [0, n.n_sp + param * n.n_p, 0], [-1, n.n_p, -1]))
        a_m = tf.stack(a_m_, axis = 2)
        a_p = tf.stack(a_p_, axis = 2)
        return a, a_m, a_p   
    
    def simulation_covariance(n, output):
        # CALCULATE THE COVARIANCE
        # called from fisher()
        # output              TF variable               - the activated output at the last layer for central simulations
        # returns covariance and mean (cov, mean)
        #______________________________________________________________
        # mean                TF variable               - mean of central simulation network outputs
        # outmm               TF variable               - difference between output and mean
        # cov                 TF variable               - unbiased covarance
        #______________________________________________________________ 
        mean = tf.reduce_mean(output, axis = 1, keepdims = True)
        outmm = tf.subtract(output, mean)
        cov = tf.divide(tf.einsum('ijk,ijl->ikl', outmm, outmm), (n.n_s - 1.))
        return cov, mean
    
    def mean_derivative(n, a_m, a_p):
        # CALCULATE THE DERIVATIVE OF THE MEAN
        # called from fisher()
        # a_m                 TF variable               - lower simulation network outputs
        # a_p                 TF variable               - upper simulation network outputs
        # returns the derivative of the mean (mean_derivative)
        #______________________________________________________________
        # derivative          TF variable               - sum of the differences divided by the difference in parameter values
        # mean_derivative     TF variable               - normalised by parameter values
        #______________________________________________________________ 
        derivative = tf.einsum('ijkl,k->ijkl', (a_p - a_m), tf.constant(n.der_den_, shape = [n.n_params], dtype = n._FLOATX))
        mean_derivative = tf.divide(tf.einsum('ijkl->ikl', derivative), n.n_p)
        return mean_derivative

    def central_values(n, a, dadv):
        # GET THE NETWORK OUTPUTS AND DERIVATIVES FROM ONLY THE CENTRAL SIMULATIONS
        # called from setup()
        # a                   list (TF tensor)                    - activated outputs at every neuron
        # dadv                list (TF tensor)                    - derivative of activated output at every neuron
        # returns the network outputs and derivatives from only the central simulations (a_, dadv_)
        #______________________________________________________________
        # a_                  list (TF tensor)                    - activated outputs at every neuron for central simulations
        # dadv_               list (TF tensor)                    - derivative of activated output for central simulations
        # l                   int                                 - layer counting variable
        #______________________________________________________________ 
        a_ = []
        dadv_ = []
        for l in range(len(n.l)):
            a_.append(tf.slice(a[l], [0, 0, 0], [-1, n.n_s, -1]))
            dadv_.append(tf.slice(dadv[l], [0, 0, 0], [-1, n.n_s, -1]))
        return a_, dadv_
    
    def backpropagate_and_update(n, W, b, a, dadv, IFI, cov, dropped):
        # BACK PROPAGATE ERROR AND UPDATE WEIGHTS
        # called from setup()
        # W                    list (TF variable)                  - neural network weights
        # b                    list (TF variable)                  - neural network biases
        # a                    list (TF tensor)                    - activated outputs at every neuron of central simulations
        # dadv                 list (TF tensor)                    - derivative of activated output of central simulations
        # IFI                  TF variable                         - determinant of the Fisher information
        # cov                  TF variable                         - covariance of network outputs      
        # dropped              list (TF placeholder)               - holder for which neurons to drop
        # returns the determinant of Fisher matrix (for plotting) and the weights and biases (IFI, W, b)
        #______________________________________________________________
        # updated_value(l, dFdb or dFdw, dropped, 'b' or 'w')  
        #                                          (dbdL or dwdL)  - Calculate the update to the weights and biases
        #______________________________________________________________
        # b_update             list (TF variable)                  - neural network bias updates
        # w_update             list (TF variable)                  - neural network weights updates
        # l                    int                                 - layer counting variable
        # IcovI                TF variable                         - determinant of the covariance matrix
        # dFdb                 TF variable                         - error in the loss with respect to the biases
        # WdFdb                TF variable                         - error in the loss wrt biases backpropagated through weights
        # dFdw                 TF variable                         - error in the loss with respect to the weights
        #______________________________________________________________
        b_update = []
        w_update = []
        for l in range(1, len(n.l)):
            if l == 1:
                IcovI = tf.matrix_determinant(cov)
                dFdb = tf.einsum('ijk,i->ijk', dadv[-l], tf.subtract(IFI, IcovI))
            else:
                WdFdb = tf.einsum('lk,ijk->ijl', W[-l + 1], dFdb)
                dFdb = tf.einsum('ijk,ijk->ijk', WdFdb, dadv[-l])
            dFdw = tf.einsum('ijk,ijl->ijkl', a[-l - 1], dFdb)
            b_update.append(n.updated_value(l, dFdb, dropped, 'b'))
            w_update.append(n.updated_value(l, dFdw, dropped, 'w'))
            b[-l] = b[-l].assign_add(b_update[-1])
            W[-l] = W[-l].assign_add(w_update[-1])
        return IFI, W, b
        
    def updated_value(n, l, derivative, dropped, which):
        # CALCULATE THE UPDATE TO THE WEIGHTS OR BIASES
        # called from backpropagate_and_update()
        # l                    int                                 - layer counting variable
        # derivative           TF variable                         - error in the loss with respect to the weights or biases 
        # dropped              list (TF placeholder)               - holder for which neurons to drop
        # which                string                              - 'b' or 'w' depending on whether to update weights or biases
        # returns the dropped out update to the weights or biases (D_dropped)
        #______________________________________________________________
        # update               TF constant                         - TensorFlow learning rate constant
        # D_batch              TF variable                         - weight or bias update for each batch
        # D                    TF variable                         - weight or bias update averaged over batch
        # D_dropped            TF variable                         - weight or bias update averaged over batch and dropped out
        #______________________________________________________________
        update = tf.multiply(tf.constant(n.lr_, shape = (), dtype = n._FLOATX), derivative)
        if which == 'b':
            D_batch = tf.divide(tf.einsum('ijk->ik', update), n.n_s)
            D = tf.divide(tf.einsum('ij->j', D_batch), n.n_batches)
        if which == 'w':
            D_batch = tf.divide(tf.einsum('ijkl->ikl', update), n.n_s)
            D = tf.divide(tf.einsum('ijk->jk', D_batch), n.n_batches)
        D_dropped = tf.multiply(D, dropped[-l])
        return D_dropped
                
    def train(n, train_data, n_epochs, test_data = None):
        # TRAIN NETWORK
        # train_data           list [3]                   - training data [central, lower, upper] simulations
        # n_epoch              int                        - number of epochs to train
        # test_data            list [3]                   - test data [central, lower, upper] simulations
        # returns the determinant of the Fisher from the training, test, output, weights and biases
        #______________________________________________________________
        # data_for_fisher(train_data, train) (data)       - selects how much data to use
        # epoch(test, train)                 (batch)      - constructs numpy array to input to the network
        # dropout()                          (dropped)    - creates the selection of neurons which get dropped
        # network_dictionary(batch, dropped, graph)
        #                                    (dictionary) - creates the dictionary used to feed the network
        # shuffle(data)                      (data)       - shuffles the order of the data
        # training(data)            (weights_and_biases)  - performs the training of the network over one epoch
        #______________________________________________________________
        # F_arr                list                       - determinant of Fisher information after training
        # train_F              list                       - determinant of Fisher information without dropout
        # test_F               list                       - determinant of Fisher information from test set
        # data                 array                      - training data
        # test                 array                      - test data
        # test_batch           array                      - test data in network format
        # d                  n float                      - value of the dropout to create dropout array
        # test_dropped         list                       - dropout list where none of the neurons are dropped
        # temp_batch           array                      - temporary training data to use to create network dictionary
        # test_dictionary      dict                       - dictionary to feed to the network when calculating training Fisher
        # epoch_ind            int                        - epoch counting variable
        # weights_and_biases   list                       - Fisher information, weights and biases [IFI, W, b]
        # train_batch          array                      - training data to use to create network dictionary
        #______________________________________________________________
        F_arr = []
        train_F = []
        test_F = []
        data = n.data_for_fisher(train_data)
        if test_data is not None:
            test = n.data_for_fisher(test_data, train = 1)
            test_batch = n.epoch(test, train = False)
        test_dropped = n.dropout(d = 0.)
        temp_batch = n.epoch(data)[-1:]
        test_dictionary = n.network_dictionary(temp_batch, test_dropped, n.training_graph)
        for epoch_ind in tqdm.tqdm(range(n_epochs)):
            data = n.shuffle(data)
            weights_and_biases = n.training(data)
            F_arr.append(weights_and_biases[0])
            train_batch = n.epoch(data)[-1:]
            test_dictionary[n.training_graph.get_tensor_by_name('x:0')] = train_batch
            train_F.append(n.train_session.run(n.IFI, feed_dict = test_dictionary))
            if test_data is not None:
                test_dictionary[n.training_graph.get_tensor_by_name('x:0')] = test_batch
                test_F.append(n.train_session.run(n.IFI, feed_dict = test_dictionary))
        return train_F, test_F, F_arr, weights_and_biases[1], weights_and_biases[2] 
    
    def data_for_fisher(n, data, train = None):
        # SELECT AMOUNT OF DATA TO USE
        # called in train()
        # data                 list [3]                   - data [central, lower, upper] simulations
        # train                int                        - amount of data to use
        # n_train            n int                        - if the training is not selected use all the data
        # returns selected data
        #______________________________________________________________     
        if train is None:
            train = n.n_train
        return [data[0][:n.n_s * train], data[1][:train * n.n_p], data[2][:train * n.n_p]]

    def epoch(n, data, train = True):
        # ARRANGES DATA INTO NETWORK FRIENDLY FORMAT
        # called in train()
        #           training()
        # data                 list [3]                   - data [central, lower, upper] simulations
        # train                bool                       - switch whether using training data or test data
        # n_train            n int                        - number of combinations to split training set into
        # n_s                n int                        - number of simulations in each combination
        # n_p                n int                        - number of differentiation simulations in each combination
        # n.n_sp             n int                        - index of simulations + lower derivatives
        # n.n_spp            n int                        - index of simulations = all derivatives
        # n.n_pp             n int                        - index of all derivatives
        # n.partial_sims     n int                        - index of partial derivatives 
        # returns data arranged for the network (epoch)
        #______________________________________________________________
        # n_train              int                        - number of combinations to split training set into
        # epoch                array                      - data arranged for the network
        # unpack_m             array                      - low simulation selection
        # unpack_p             array                      - high simulation selection
        # param                int                        - parameter derivative counter
        # low                  int                        - lower index selector
        # high                 int                        - higher index selector
        #______________________________________________________________
        if train:
            n_train = n.n_train
        else:
            n_train = 1
        epoch = np.zeros([n_train, n.n_spp, n.l[0]])
        for train in range(n_train):
            epoch[train, : n.n_s] = data[0][train: n.tot_sims: n_train]
            unpack_m = np.zeros([n.n_pp, n.l[0]])
            unpack_p = np.zeros([n.n_pp, n.l[0]])
            for param in range(n.n_params):
                low = param * n.n_p
                high = low + n.n_p
                unpack_m[low: high] = data[1][train: n.partial_sims: n_train, param]
                unpack_p[low: high] = data[2][train: n.partial_sims: n_train, param]    
            epoch[train, n.n_s: n.n_sp] = unpack_m
            epoch[train, n.n_sp:] = unpack_p
        return epoch
    
    def dropout(n, d = None):
        # CREATES THE DROPOUT ARRAY
        # called in train()
        #           training()
        # d                    float                      - value of the dropout to create dropout array

        # l                  n list                       - contains the neural architecture of the network
        # d                  n float                      - value of the dropout to create dropout array
        # returns array of the dropped out neurons (dropped)
        #______________________________________________________________
        # l                    int                        - layer counting variable
        # dropped              list                       - list of dropped out neurons
        # dropout              array                      - whether to drop a neuron or not
        # accept               array                      - acceptance probability of whether to drop a neuron or not
        #______________________________________________________________
        if d is None:
            d = n.d
        dropped = []
        for l in range(len(n.l) - 1):
            dropout = np.ones(n.l[l + 1])
            if d == 0.:
                dropped.append(dropout)
            else:
                if not ((n.l[-1] == 1) and (l == len(n.l) - 2)):
                    accept = np.random.uniform(0., 1., size = (n.l[l + 1]))
                    dropout[accept <= d] = 0.
                dropped.append(dropout)
        return dropped
    
    def training(n, data):
        # ACTUAL TRAINING ROUTINE
        # called in train()
        # data                  array      - training data in network friendly format
        # n_train             n int        - number of combinations to split training set into
        # n_batches           n int        - number of batches to calculate at one time
        # training_graph      n TF obj     - TensorFlow graph for training the network
        # train_session       n TF session - initialised TensorFlow session for training
        # returns the Fisher determinant and weights and biases (weights_and_biases)
        #______________________________________________________________
        # ceil(n_train/n_batches)   n u (int)             - calculates the ceiling value of how many batches to train over
        # epoch(test, train)                 (batch)      - constructs numpy array to input to the network
        # dropout()                          (dropped)    - creates the selection of neurons which get dropped
        # network_dictionary(batch, dropped, graph)
        #                                    (dictionary) - creates the dictionary used to feed the network
        #______________________________________________________________
        # train_ind             int        - batch loop counting variable
        # batch_ind             int        - index of batch selection
        # dropped               list       - list of dropped out neurons
        # dictionary            dict       - dictionary to feed to the network
        # weights_and_biases    list       - Fisher information, weights and biases [IFI, W, b]
        #______________________________________________________________
        for train_ind in range(n.u.ceil(n.n_train / n.n_batches)):
            batch_ind = train_ind * n.n_batches
            train_batch = n.epoch(data)[batch_ind: batch_ind + n.n_batches:]
            if train_batch.shape[0] == n.n_batches:
                dropped = n.dropout()
                dictionary = n.network_dictionary(train_batch, dropped, n.training_graph)
                weights_and_biases = n.train_session.run(n.weights_and_biases, feed_dict = dictionary)
        return weights_and_biases

    def network_dictionary(n, data, dropped, graph, train = None):
        # CREATES DICTIONARY TO FEED THE NETWORK
        # called in train()
        #           training()
        # data                  array      - data in network friendly format
        # dropped               list       - list of dropped out neurons
        # graph               n TF obj     - TensorFlow graph describing the network
        # train                 list       - weights and biases to set the network with when not training
        # l                   n list       - contains the neural architecture of the network
        # returns dictionary to feed the network (dictionary)
        #______________________________________________________________
        # dictionary            dict       - dictionary to feed to the network
        # l                     int        - layer counting variable
        #______________________________________________________________
        dictionary = {}
        dictionary[graph.get_tensor_by_name('x:0')] = data
        for l in range(len(n.l) - 1):
            dictionary[graph.get_tensor_by_name('d_' + str(l) + ':0')] = dropped[l]
            if train is not None:
                dictionary[graph.get_tensor_by_name('W_' + str(l) + ':0')] = train[0][l]
                dictionary[graph.get_tensor_by_name('b_' + str(l) + ':0')] = train[1][l]
        return dictionary
    
    def shuffle(n, data):
        # SHUFFLES THE DATA
        # called in train()
        # data                 list        - training data
        # tot_sims           n int         - total number of simulations to use
        # returns the shuffled data (data)
        #______________________________________________________________
        # total_shuffle        array       - array of indices to shuffle with
        # partial_shuffle      array       - array of indices to shuffle the derivatives with
        #______________________________________________________________
        total_shuffle = np.arange(n.tot_sims)
        np.random.shuffle(total_shuffle)
        partial_shuffle = np.arange(n.partial_sims)
        np.random.shuffle(partial_shuffle)
        data[0] = data[0][total_shuffle]
        data[1] = data[1][partial_shuffle]
        data[2] = data[2][partial_shuffle]
        return data
        
    def session(n, graph):
        # INITIALISE THE TENSORFLOW SESSION
        # called in setup()
        # sess      n TF session   - initialised TensorFlow session
        # returns the session (sess)
        #______________________________________________________________
        sess = tf.InteractiveSession(graph = graph)
        sess.run(tf.global_variables_initializer())
        return sess
    
    def resetup(n, trained_W, trained_b):
        # REBUILD AN EMPTY NETWORK TO USE PRETRAINED WEIGHTS AND BIASES
        # trained_W   list       - trained neural network weights
        # trained_b   list       - trained neural network biases
        # l         n list       - contains the neural architecture of the network
        # returns a TensorFlow session and graph and a dictionary with no data, but zero dropout
        #______________________________________________________________
        # feedforward(x, W, b, dropped)   (a, dadv)            - feedforward inputs to get outputs and derivatives
        #______________________________________________________________
        # graph    n TF obj                  - TensorFlow graph for training the network
        # W          list (TF placeholder)   - holder for neural network weights
        # b          list (TF placeholder)   - holder for neural network biases
        # dropped    list (TF placeholder)   - holder for which neurons to drop
        # x          list (TF placeholder)   - holder for network input
        # l          int                     - layer counting variable
        # a        n list (TF tensor)        - activated outputs at every neuron
        # dadv       list (TF tensor)        - derivative of activated output at every neuron
        # session    TF session              - initialised TensorFlow session
        # graph      TF obj                  - TensorFlow graph of pretrained network
        # dictionary            dict       - dictionary to feed to the network
        #______________________________________________________________
        graph = tf.Graph()
        with graph.as_default() as g:
            W = []
            b = []
            x = tf.placeholder(n._FLOATX, shape = [1, 1, n.l[0]], name = 'x')
            dropped = []
            for l in range(len(n.l) - 1):
                W.append(tf.placeholder(n._FLOATX, shape = [n.l[l], n.l[l + 1]], name = 'W_' + str(l)))
                b.append(tf.placeholder(n._FLOATX, shape = [n.l[l + 1]], name = 'b_' + str(l)))
                dropped.append(tf.placeholder(n._FLOATX, shape = [n.l[l + 1]], name = 'd_' + str(l)))
            n.a, dadv = n.feedforward(x, W, b, dropped)
        session = n.session(graph)
        dropped = n.dropout(d = 0.)
        dictionary = n.network_dictionary(None, dropped, graph, train = [trained_W, trained_b])
        return session, graph, dictionary 

class test_models():
    def __init__(t, parameters):
        # INITIALISE TEST MODEL
        # parameters dict              - loads parameters into Gaussian noise, Lyman-α model, or LISA model
        # 'method'           string    - 'Gaussian', 'Lyman-α', or 'LISA'
        # 'fiducial θ'       float     - parameter to create data for training network
        # 'total number of simulations' 
        #                    int       - simulations to make for training network
        # 'derivative'       list      - list of lower and upper amount to vary fiducial θ by to get derivative
        # 'number of inputs' int       - (Gaussian) number of inputs to network
        # 'noise'            None, float or list - (Gaussian) amount of noise or prior on noise
        # 'bin_size'         float     - (Lyman) bin resolution for quasar spectrum
        # 'z'                float     - (Lyman) redshift of the quasar
        # 'cosmology'        dict      - (Lyman) dictionary with astropy cosmology
        # 'H_0'              float     - Hubble constant
        # 'Ω_m'              float     - critical density of matter 
        # 'Ω_b'              float     - critical density of baryons
        # 'σ_8'              float     - amplitude of pertubations 8h^{-1}Mpc spheres
        # 'n_s'              float     - spectral tilt
        # 'm_ν'              list      - mass of neutrinos
        # 'N_eff'            float     - number of degrees of freedom in radiation
        # 'T_CMB'            float     - temperature of the CMB
        # 't_L'              float     - light travel time along LISA arm
        # 'S_acc'            float     - proof mass acceleration noise
        # 'S_sn'             float     - LISA shot noise
        # 'Q'                float     - width of the gravitational waveform
        # 't_c'              float     - time of the gravitational event
        # 'SN'               float     - signal to noise of gravitational event
        #______________________________________________________________
        # P1D()             ()         - calculates power spectrum and bin resolution
        # S_h(params)       ()         - calculates one-sided noise spectral density 
        #______________________________________________________________
        # u                t class     - load utilities
        # method           t func      - t.Gaussian() or t.random_field()
        # θ                t float     - parameter to create data for training network
        # noise            t float or list - amount of noise or prior on noise
        # inputs           t int       - number of inputs to network
        # bin_size         t float     - bin resolution for quasar spectrum
        # z                t float     - redshift of the quasar
        # cosmology        t dict      - dictionary with astropy cosmology
        # 'H_0'              float     - Hubble constant
        # 'Ω_m'              float     - critical density of matter 
        # 'Ω_b'              float     - critical density of baryons
        # 'σ_8'              float     - amplitude of pertubations 8h^{-1}Mpc spheres
        # 'n_s'              float     - spectral tilt
        # 'm_ν'              list      - mass of neutrinos
        # 'N_eff'            float     - number of degrees of freedom in radiation
        # 'T_CMB'            float     - temperature of the CMB
        # t_L              t float     - light travel time along LISA arm
        # S_acc            t float     - proof mass acceleration noise
        # S_sn             t float     - LISA shot noise
        # Q                t float     - width of the gravitational waveform
        # A                t float     - amplitude of gravitational waveform
        # t_c              t float     - time of the gravitational event
        # SN               t float     - signal to noise of gravitational event
        # tot_sims         t int       - total number of simulations to use
        #______________________________________________________________
        t.u = utils()
        if parameters['method'] == 'Gaussian':
            t.method = t.Gaussian
            t.θ = parameters['fiducial θ']
            _ = t.u.check_error(t.θ, [float, operator.le, 0.], 'Fiducial variance must be a positive float')
            if parameters['noise'] is None:
                t.noise = 0.
            elif type(parameters['noise']) == float:
                t.noise = parameters['noise']
            elif (type(parameters['noise']) == list):
                if len(parameters['noise']) == 2:
                    t.noise = parameters['noise']
                else:
                    print('If the noise is lower and upper prior eges the length needs to be 2')
                    sys.exit()
            else:
                print('The noise needs to be a float or a list with the lower and upper bounds')
                sys.exit()
            t.inputs = parameters['number of inputs']
            _ = t.u.check_error(t.inputs, [int, operator.le, 0], 'Number of inputs must be a positive integer')  
        elif parameters['method'] == 'Lyman-α':
            t.method = t.random_field
            t.θ = parameters['fiducial θ']
            _ = t.u.check_error(t.θ, [float], 'Fiducial parameter must be a float') 
            t.bin_size = parameters['bin size']
            _ = t.u.check_error(t.bin_size, [float, operator.le, 0.], 'The bin size must a positive float')
            t.z = parameters['z']
            _ = t.u.check_error(t.z, [float, operator.le, 0.], 'The redshift must a positive float')
            if parameters['cosmology'] is None:
                t.cosmology = {'H_0': 67.7, 'Ω_m': 0.307, 'Ω_b': 0.0468, 'σ_8': 0.8159, 'n_s': 0.9667, 'm_ν': [0., 0., 0.06], 'N_eff': 3.05, 'T_CMB': 2.725 }
            else:
                t.cosmology = parameters['cosmology']
            t.P1D()
        elif parameters['method'] == 'LISA':
            t.method = t.gravitational_wave_burst
            t.θ = parameters['fiducial θ']
            _ = t.u.check_error(t.θ, [float, operator.le, 1e-3, operator.ge, 0.5], 'Fiducial parameter must be a positive float between 1e-3<θ<0.5')
            t.t_L = parameters['t_L']
            _ = t.u.check_error(t.t_L, [float, operator.le, 0.], 'Light travel time along LISA arm should be a positive float')
            t.S_acc = parameters['S_acc']
            _ = t.u.check_error(t.S_acc, [float, operator.le, 0.], 'Proof mass acceleration noise should be a positive float')
            t.S_sn = parameters['S_sn']
            _ = t.u.check_error(t.S_sn, [float, operator.le, 0.], 'Shot noise should be a positive float')
            t.Q = parameters['Q']
            _ = t.u.check_error(t.Q, [float, operator.le, 0.], 'The gravitational waveform width should be a positive float')
            t.A = parameters['A']
            _ = t.u.check_error(t.A, [float, operator.le, 0.], 'The gravitational waveform amplitude should be a positive float')
            t.t_c = parameters['t_c']
            _ = t.u.check_error(t.t_c, [float, operator.le, 0.], 'The burst time should be a positive float')
            t.SN = parameters['SN']
            _ = t.u.check_error(t.SN, [float, operator.le, 0.], 'Fiducial parameter must be a positive float')
            t.detector_noise_power()
        else:
            print('The method must be one of Gaussian or Lyman-alpha.')
            sys.exit()
        t.tot_sims = parameters['total number of simulations']
        _= t.u.check_error(t.tot_sims, [int, operator.le, 0], 'Total number of simulations must be a positive integer')
        t.derivative = parameters['derivative']
              
    def create_data(t, derivative = False, θ = None, noise = None, num = None, shaped = False):
        # INITIAL CREATION OF DATA
        # derivative    bool          - whether to calculate derivatives
        # θ             None, float   - central parameter value
        # noise         None, float or list - noise for the Gaussian model
        # num           None, int     - number of simulations to make
        # tot_sims    t int           - total number of simulations to make
        # method      t function      - the function to generate the data
        # θ           t float         - fiducial parameter value
        # returns either the data or a list of the data and the derivatives (data or [data, data_m, data_p])
        #______________________________________________________________
        # get_noise(noise, num) ()    - generates noise for Gaussian noise
        # Gaussian(θ, simulation, random_seed)
        #                   (data)    - generates the Gaussian noise
        # random_field(θ, simulation, random_seed)
        #                   (data)    - generates the quasar spectrum
        # derivative_denominator(θ) ()- calculates the denominator of the derivative
        #______________________________________________________________
        # data          array         - generated data
        # seed          int           - random seed
        # low_θ         float         - lower parameter simulation
        # data_m        array         - data at the lower parameter value
        # high_θ        float         - higher parameter simulation
        # data_p        array         - data at the higher parameter value
        #______________________________________________________________
        if num is None:
            num = t.tot_sims
        if t.method == t.Gaussian:
            t.get_noise(noise, num)
        if θ is None:
            θ = t.θ
        data = np.array([t.method(θ, simulation) for simulation in range(num)])
        if derivative:
            seed = np.random.randint(1e8)
            if t.method == t.Gaussian:
                t.get_noise(noise, num)
            t.derivative_denominator(θ)
            data_m = np.swapaxes(np.array([[t.method(t.derivative[0], simulation, random_seed = seed + simulation) for simulation in range(num)]]), 0, 1)
            data_p = np.swapaxes(np.array([[t.method(t.derivative[1], simulation, random_seed = seed + simulation) for simulation in range(num)]]), 0, 1)
            np.random.seed()
            return [data, data_m, data_p]
        else:
            if shaped:
                return data[0][np.newaxis, np.newaxis, :]
            else:
                return data

    def get_noise(t, noise, num):
        # GENERATES THE NOISE FOR THE GAUSSIAN TEST
        # called in create_data()
        #           ABC()
        #           PMC()
        # noise       None, float or list  - either the value of the noise or lower and upper bounds on the prior of the noise
        # num         int                  - length of the noise array
        # noise     t None, float or list  - either the value of the noise or lower and upper bounds on the pior of the noise
        #______________________________________________________________ 
        # noise_arr t array                - array with the random noise for Gaussian model 
        #______________________________________________________________ 
        if noise is None:
            noise = t.noise
        if type(noise) == float:
            t.noise_arr = np.ones(num) * noise
        elif (type(noise) == list):
            t.noise_arr = np.random.uniform(noise[0], noise[1], num)
    
    def derivative_denominator(t, θ):
        # CALCULATES THE DENOMINATOR FOR THE NUMERICAL DERIVATIVE
        # called in create_data()
        # θ            float       - fiducial value for the data
        # derivative t float       - amount of deviation from the fiducial value
        #______________________________________________________________ 
        # der_den    t array       - denominator for the numercial derivatives 
        #______________________________________________________________ 
        t.der_den = np.array([1. / (t.derivative[1] - t.derivative[0])])
        #if t.method == t.Gaussian:
        #    t.der_den = np.array([1. / (2. * t.derivative)])
        #elif t.method == t.random_field:
        #    t.der_den = np.array([1. / (np.exp(θ + t.derivative) - np.exp(θ - t.derivative))])
        #elif t.method == t.gravitational_wave_burst:
        #    t.der_den = np.array([1. / (np.exp(θ + t.derivative) - np.exp(θ - t.derivative))])
        
    def Gaussian(t, θ, simulation, random_seed = None, shaped = False):
        # CREATES A SINGLE SIMULATION OF GAUSSIAN NOISE
        # called in create_data()
        #           ABC()
        #           PMC()
        # θ             float         - central parameter value
        # simulation    int           - simulation intialisation index
        # random_seed   int           - random seed to spawn the noise with
        # shaped        bool          - whether to return a shaped array or not
        # noise_arr   t array         - array with the random noise for Gaussian model 
        # inputs      t int           - number of data points in the simulation
        # returns a single simulation of Gaussian noise (gaussian)
        #______________________________________________________________ 
        # gaussian      array         - Gaussian noise for test model
        #______________________________________________________________ 
        if random_seed is not None:
            np.random.seed(random_seed)
        gaussian = np.random.normal(0., np.sqrt(θ + t.noise_arr[simulation]), t.inputs)
        if shaped:
            return gaussian[np.newaxis, np.newaxis, :]
        else:
            return gaussian

    def random_field(t, θ, _, random_seed = None, shaped = False, bias_calculator = None):
        # CREATES A SINGLE SIMULATION OF QUASAR SPECTRUM
        # called in create_data()
        #           ABC()
        #           PMC()
        # θ                   float   - central parameter value
        # _                   None    - placeholder for function input
        # random_seed         int     - random seed to spawn the noise with
        # shaped              bool    - switch whether to return a shaped array or not
        # bias_calculator     bool    - whether to calculate 
        # NF                t int     - number of frequency bins
        # P_F               t array   - power spectrum of flux absorbing field
        # N                 t int     - number of real space bins
        # L                 t float   - size of survey
        # γ                 t float   - temperature density relation exponent
        # τ_factor          t float   - Gunn Peterson approximate optical depth
        # λ_                t array   - wavelength values in observer rest frame
        # λ_min_true_smooth t float   - minimum wavelength in observer rest frame 
        # λ_max_true_smooth t float   - maximum wavelength in observer rest frame
        # C                 t spline  - value of the continuum flux at wavelengths
        # λ___              t array   - wavelength values over observed region
        # bins              t list    - indices to sum the flux over to get binned data
        # inputs            n int     - number of data points in the simulation
        # returns a single simulation of Gaussian noise (poisson)
        #______________________________________________________________ 
        # poisson(fluxb, random_seed)  (poisson)  - Applies poisson noise to the flux
        #______________________________________________________________
        # noise               array   - complex noise for random Gaussian field
        # modes               array   - random field modulated by power spectrum
        # δ_F                 array   - density field
        # ρ                   array   - mean density field
        # τ                   array   - optical depth
        # F                   array   - quasar flux
        # f                   array   - continuum modulated quasar flux
        # fluxb               array   - binned continuum modulated quasar flux
        # poisson             array   - binned continuum modulated quasar flux in photon counts with Poisson noise
        if random_seed is not None:
            np.random.seed(random_seed)
        noise = np.random.normal(0., 1., t.NF) + np.random.normal(0., 1., t.NF) * 1j
        modes = np.empty(t.NF, dtype = complex)
        if bias_calculator is not None:
            P_F = bias_calculator * t.P_F
        else:
            P_F = t.P_F
        modes[0] = np.sqrt(np.exp(θ) * P_F[0]) * noise[0].real
        modes[-1] = np.sqrt(np.exp(θ) * P_F[-1]) * noise[-1].real
        modes[1: -1] = noise[1: -1] * np.sqrt(0.5 * np.exp(θ) * P_F[1: - 1])
        δ_F = np.fft.irfft(modes) * t.N / t.L / 2.
        ρ = np.exp(δ_F)
        ρ /= np.mean(ρ)
        τ = t.τ_factor * ρ**(2. - 0.7 * t.γ)
        F = np.exp(-τ)[(t.λ_ >= t.λ_min_true_smooth) * (t.λ_ <= t.λ_max_true_smooth)]
        if bias_calculator is not None:
            return F
        f = F * t.C(t.λ___)
        fluxb = np.array([np.mean(f[t.bins[i]: t.bins[i + 1]]) for i in range(t.inputs)]).astype(int)
        poisson = t.poisson(fluxb, random_seed)
        if shaped:
            return poisson[np.newaxis, np.newaxis, :]
        else:
            return poisson
    
    def P1D(t, bin_size = None, z = None):
        # CALCULATES THE POWER SPECTRUM OF FLUX DENSITY AND BINS
        # called in __init__()
        # bin_size            None, float   - bin resolution for quasar spectrum
        # z                   None, float   - redshift of the quasar
        # cosmology         t dict          - dictionary with astropy cosmology
        # 'H_0'               float         - Hubble constant
        # 'Ω_m'               float         - critical density of matter 
        # 'Ω_b'               float         - critical density of baryons
        # 'σ_8'               float         - amplitude of density pertubations in spheres of 8h^{-1}Mpc
        # 'n_s'               float         - spectral tilt
        # 'm_ν'               list          - mass of neutrions
        # 'N_eff'             float         - number of degrees of freedom in radiation
        # 'T_CMB'             float         - temperature of the CMB
        # returns a single simulation of Gaussian noise (poisson)
        #______________________________________________________________ 
        # astropy.cosmology                 - external astropy module
        # hmf.transfer                      - external hmf module
        # si.InterpolatedUnivariateSpline   - external scipy.interpolate interpolation
        # create_bins(n, λ, size, log = True)   
        #       (λ_centres, λ_widths, bins) - find the bins for the binned quasar spectrum
        #______________________________________________________________
        # cosmo               astropy obj   - astropy Flat LambdaCDM cosmology 
        # h                   float         - dimensionless Hubble constant
        # NF                t int           - number of frequency bins
        # N                 t int           - number of real space bins
        # r                   array         - real space array for integration
        # R                   float         - smoothing scale
        # ξ                   array         - real space correlation function
        # z_min               float         - minimum redshift
        # z_max               float         - maximum redshift
        # start               float         - closest survey point
        # end                 float         - furthest survey point
        # L                 t float         - size of survey
        # dk                  float         - frequency steps
        # k                   array         - frequency array
        # k_par               array         - k along the line of sight
        # μ_k                 array         - angle between line of sight and frequency
        # b_δ                 float         - bias parameter
        # β                   float         - flux modification parameter
        # k_NL                float         - small scale structure constant
        # α_NL                float         - small scale structure constant
        # k_P                 float         - small scale structure constant
        # α_P                 float         - small scale structure constant
        # k_V0                float         - small scale structure constant
        # k_Vp                float         - small scale structure constant
        # α_Vp                float         - small scale structure constant
        # α_V                 float         - small scale structure constant
        # k_V                 float         - small scale structure constant
        # D_F                 array         - small scale structure modification to power spectrum
        # α                   float         - redshift modification constant
        # P_F               t array         - power spectrum of flux absorbing field
        # A                   float         - Gunn Peterson flux amplitude
        # T0                  float         - temperature density relation constant
        # γ                 t float         - temperature density relation exponent
        # Γ_UV                float         - photon ionisation value
        # τ_factor          t float         - fluctating Gunn Peterson approximate optical depth
        # pca                 array         - PCA components for continuum
        # λ_pca               array         - PCA wavelengths
        # μ_pca               array         - PCA amplitude means
        # σ_pca               array         - PCA amplitude standard deviations
        # ξ_pca               array         - PCA components
        # c_pca               array         - PCA coefficients
        # r_                  array         - continuum flux at quasar rest frame     
        # λ_α                 float         - Lyman-α peak
        # λ_min_ind_smooth    ind           - index of minimum wavelength
        # λ_max_ind_smooth    ind           - index of maximum wavelength
        # λ_true              array         - wavelengths in observer rest frame
        # λ_min_true_smooth t float         - minimum wavelength in observer rest frame 
        # λ_max_true_smooth t float         - maximum wavelength in observer rest frame
        # C                 t spline        - value of the continuum flux at values of the wavelength
        # x                   array         - real space value for interpolation
        # z_                  array         - redshifts for interpolation
        # s                   array         - comoving distance at redshift values
        # sz                  spline        - spline of the comoving distance at given redshift things
        # λ_                t array         - wavelength values in observer rest frame
        # λ___              t array         - wavelength values over observed region
        # λ_centres         t array         - centre wavelengths of the bins
        # λ_widths          t array         - widths of the wavelength bins
        # bins              t list          - indices to sum the flux over to get binned data
        # num_centres       t               - number of bin edges
        # inputs            t int           - number of inputs to the network
        import hmf
        import scipy.interpolate as si
        import astropy
        if z is None:
            z = t.z
        cosmo = astropy.cosmology.FlatLambdaCDM(H0 = t.cosmology['H_0'], Om0 = t.cosmology['Ω_m'], Ob0 = t.cosmology['Ω_b'], m_nu = t.cosmology['m_ν'] * astropy.units.eV, Neff = t.cosmology['N_eff'], Tcmb0 = t.cosmology['T_CMB'])
        T = hmf.transfer.Transfer(dlnk = 0.005, transfer_model = 'EH', z = 2.25, sigma_8 = t.cosmology['σ_8'], n = t.cosmology['n_s'], cosmo_model = cosmo)
        h = cosmo.H0.value / 100.
        t.NF = 8192
        t.N = int(2 * (t.NF - 1))
        r = np.linspace(-200, 200, t.NF)
        R = 5.
        ξ = np.array([np.trapz(np.exp(-R**2. * T.k**2.) * T.k**2. * T.power * np.sinc((T.k * r[i]) / np.pi) / 2. / np.pi**2., x = T.k) for i in range(t.NF)])
        z_min = 1.96
        z_max = 3.44
        start = T.cosmo.comoving_distance(z_min).value * T.cosmo.h
        end = T.cosmo.comoving_distance(z_max).value * T.cosmo.h
        t.L = (end - start)
        dk = 1. / t.L
        k = np.linspace(0.001, dk * t.N, t.NF)
        k_par = k
        μ_k = k_par / k
        k_NL = 6.40
        α_NL = 0.569
        k_P = 15.3
        α_P = 2.01
        k_V0 = 1.220
        k_Vp = 0.923
        α_Vp = 0.451
        α_V = 1.50
        k_V = k_V0 * (1. + k / k_Vp)**α_Vp
        D_F = np.exp((k / k_NL)**α_NL - (k / k_P)**α_P - (k_par / k_V)**α_V)
        α = 3.2
        t.P_F = t.L * np.array([np.trapz(np.exp(1j * k[i] * r) * ξ, x = r) for i in range(t.NF)]) * D_F
        t.P_F[t.P_F < 0] = 0.
        A = 1.54 
        T0 = 18400
        t.γ = 0.29
        Γ_UV = 4e-12
        t.τ_factor = A * (T0 / 1e4)**-0.7 * (1e-12 / Γ_UV) * ((1. + z) / (1. + 3.))**6. * (0.7 / h) * (cosmo.Ob0 * h**2. / 0.02156)**2. * (4.0927 / (cosmo.H(z).value / cosmo.H0.value))
        pca = np.loadtxt('data/continuum.txt')
        λ_pca = pca[:, 0] / 10.
        μ_pca = pca[:, 1]
        σ_pca = pca[:, 2]
        ξ_pca = pca[:, 3:]
        c_pca = 0.4 * np.ones(ξ_pca.shape)
        r_ = μ_pca + np.sum(c_pca * ξ_pca)
        λ_α = 121.567
        λ_min_ind_smooth = np.argmin(np.abs(λ_pca - 104.))
        λ_max_ind_smooth = np.argmin(np.abs(λ_pca - 120.))
        λ_true =  λ_pca * (1. + z)
        t.λ_min_true_smooth = λ_true[λ_min_ind_smooth]
        t.λ_max_true_smooth = λ_true[λ_max_ind_smooth]
        t.C = si.InterpolatedUnivariateSpline(λ_true, r_)
        x = np.linspace(start, end, num = t.N)
        z_ = np.linspace(0, 10)
        s = T.cosmo.comoving_distance(z_).value * T.cosmo.h
        sz = si.InterpolatedUnivariateSpline(s, z_)
        t.λ_ = λ_α * (1. + sz(x))
        t.λ___ = t.λ_[(t.λ_ >= t.λ_min_true_smooth) * (t.λ_ <= t.λ_max_true_smooth)]
        if bin_size is None:
            bin_size = t.bin_size
        t.λ_centres, t.λ_widths, t.bins = t.create_bins(t.λ___, bin_size)
        t.num_centres = len(t.λ_centres)
        t.inputs = len(t.bins) - 1
        t.β = so.fsolve(t.bias_finder, 0.)[0]
        t.P_F = t.P_F * t.β
    
    def bias_finder(t, β):
        return 0.8 - np.mean(t.random_field(t.θ, None, random_seed = 0, bias_calculator = β))
            
    def create_bins(t, λ, size, log = True):
        # FIND THE BINS FOR THE QUASAR SPECTRUM
        # called in P1D()
        # λ            array       - wavelength array
        # size         float       - bin resolution
        # log          bool        - whether to use resolution in logspace
        # returns the wavelengths, their widths and the bins to sum over
        #______________________________________________________________ 
        # find_bins(λ, bins[-1], size, log) (bin)      - finds the next bin in to create the resolution at the correct size
        #______________________________________________________________
        # bins        list         - bins to sum over to get binned flux
        # λ_edges     list         - edges of the bins in wavelength space
        # λ_centres   list         - centres of the bins in wavelength space
        # λ_widths    list         - width of the bins in wavelength space
        #______________________________________________________________
        bins = [0]
        λ_edges = [λ[bins[-1]]]
        λ_centres = []
        λ_widths = []
        while bins[-1] < len(λ):
            bins.append(t.find_bins(λ, bins[-1], size, log))
            if bins[-1] == len(λ):
                λ_edges.append(λ[-1])
            else:
                λ_edges.append(λ[bins[-1]])
            λ_centres.append(λ_edges[-2] + (λ_edges[-1] - λ_edges[-2]) / 2.)
            λ_widths.append(λ_edges[-1] - λ_edges[-2])
        return np.array(λ_centres), np.array(λ_widths), bins 
    
    def find_bins(t, λ, start, size, log):
        # FIND THE INDEX OF THE NEXT BIN AT A GIVEN RESOLUTION
        # called in create_bins()
        # λ            array       - wavelength array
        # start        int         - index of first bin edge
        # size         float       - bin resolution
        # log          bool        - whether to use resolution in logspace
        # returns the index of the next bin edge (ind)
        #______________________________________________________________ 
        # step(ind, λ, log, start) (step_size)   - Finds the size of the step in wavelength space
        #______________________________________________________________
        # ind          int         - index of the wavelength array to calculate the resolution
        #______________________________________________________________
        ind = start + 1
        while(t.step(ind, λ, log, start) <= size):
            ind += 1
            if ind == len(λ):
                return ind
        return ind
    
    def step(t, ind, λ, log, start):
        # FIND THE SIZE OF THE STEP BETWEEN BINS IN WAVELENGTH SPACE
        # called in find_bins()
        # ind          int         - index of furthest bin edge
        # λ            array       - wavelength array
        # log          bool        - whether to use resolution in logspace
        # start        int         - index of first bin edge
        # returns the step size between two wavelengths
        #______________________________________________________________ 
        if log:
            return np.log10(λ[ind]) - np.log10(λ[start])
        else:
            return λ[ind] - λ[start]
    
    def poisson(t, f, seed):
        # APPLY GAUSSIAN NOISE TO QUASAR SPECTRUM
        # called in random_field()
        # f              array      - continuum modulated flux
        # seed           int        - random seed for the poisson noise
        # returns the quasar spectrum with Poisson noise (poisson)
        #______________________________________________________________ 
        # temp_seed      int        - random seed for each element of array of the poisson noise
        # poisson        array      - quasar spectrum with Poisson noise
        #______________________________________________________________
        if seed is not None:
            temp_seed = np.random.randint(1e6)
            poisson = []
            for ind in range(len(f)):
                np.random.seed(seed + ind + temp_seed)
                poisson.append(np.random.poisson(f[ind]))
            poisson = np.array(poisson)
        else:
            poisson = np.random.poisson(f)
        return poisson
       
    def detector_noise_power(t):
        # GENERATE LISA DETECTOR NOISE SPECTRAL DENSITY
        # called in __init__()
        # t_L               t float         - light travel time along LISA arm
        # S_acc             t float         - proof mass acceleration noise
        # S_sn              t float         - LISA shot noise
        #______________________________________________________________ 
        # N                 t int           - number of real space bins
        # NF                t int           - number of frequency bins
        # inputs            t int           - number of inputs to the network
        # t                 t array         - time array for plots
        # df                  float         - frequency steps
        # f                 t array         - frequency array
        # tπftL               array         - triogometric entry (phase)
        # S_h               t array         - LISA detector noise spectral density
        #______________________________________________________________ 
        t.N = 2048
        t.NF = int(t.N/2 + 1)
        t.inputs = t.N
        t.t = np.arange(9.9e4, 9.9e4 + t.N)
        df = 1. / (2. * t.N)
        t.f = np.linspace(1e-3, df * t.N, t.NF)
        tπftL = 2. * np.pi * t.f * t.t_L
        t.S_h = 16. * np.sin(tπftL)**2. * (2. * (1. + np.cos(tπftL) + np.cos(tπftL)**2.) * (1. + (1e-4 / t.f)**2.) * t.S_acc / t.f**2. + (1. + np.cos(tπftL) / 2.) * t.S_sn * t.f**2.)   
       
    def gravitational_wave_burst(t, θ, _, random_seed = None, shaped = False):
        # CREATES A SINGLE GRAVITATIONAL WAVE BURST
        # called in create_data()
        #           ABC()
        #           PMC()
        # θ                   float   - central parameter value
        # _                   None    - placeholder for function input
        # random_seed         int     - random seed to spawn the noise with
        # shaped              bool    - switch whether to return a shaped array or not
        # NF                t int     - number of frequency bins
        # S_h               t array   - LISA detector noise spectral density
        # SN                t float   - signal to noise of the detection
        # returns a single simulation of a gravitational wave burst (s)
        #______________________________________________________________
        # noise               array   - complex noise for random Gaussian field
        # modes               array   - random field modulated by power spectrum
        # S_h                 array   - LISA detector noise in real space
        # h                   array   - Fourier space gravitational wave signature
        # s                   array   - simulation of gravitational wave (with noise)
        #______________________________________________________________
        if random_seed is not None:
            np.random.seed(random_seed)
        noise = np.random.normal(0., 1., t.NF) + np.complex(0., 1.) * np.random.normal(0., 1., t.NF)
        modes = np.empty(t.NF, dtype = complex)
        modes[0] = np.sqrt(t.S_h[0]) * noise[0].real
        modes[-1] = np.sqrt(t.S_h[-1]) * noise[-1].real
        modes[1: -1] = noise[1: -1] * np.sqrt(0.5 * t.S_h[1: -1])
        h = t.gravitational_waveform(θ)
        modes = modes / np.max(np.abs(modes)) * np.max(np.abs(h))
        s = (np.fft.irfft(t.SN * h) + np.fft.irfft(modes)).real
        if shaped:
            return s[np.newaxis, np.newaxis, :]
        else:
            return s

    def gravitational_waveform(t, θ):
        # CALCULATES GRAVITATIONAL WAVEFORM AT CENTRAL FRENQUENCY
        # called in gravitational_wave_burst()
        #           lnL_grav()
        # θ                   float   - central parameter value
        # Q                 t float   - width of gravitational wave
        # A                 t float   - amplitude of gravitational wave
        # t_c               t float   - time of gravitational wave
        # f                 t array   - frequency array
        # returns a real space gravitational waveform
        #______________________________________________________________
        # h                   array   - Fourier space gravitational waveform
        #______________________________________________________________
        h = t.A * t.Q / t.f * np.exp(-0.5 * t.Q**2. * ((t.f - θ) / θ)**2.) * np.exp(2. * np.pi * np.complex(0., 1.) * t.t_c * t.f)
        return h

    def ABC(t, W, b, F, real_data, prior, num_thetas, noise = None, real_summary = False, PMC = False, log = False):
        # APPROXIMATE BAYESIAN COMPUTATION WITH RANDOM DRAWS FROM PRIOR
        # W              list                - list of all the pretrained weights of the network
        # b              list                - list of all the pretrained biases of the network
        # F              float               - Fisher information for calculating distance
        # real_data      array               - real data to do network comparison with
        # prior          list                - lower and upper bounds of uniform prior
        # num_thetas     int                 - number of random draws from the prior
        # noise          None, list or float - noise for Gaussian model (None use preloaded noise)
        # real_summary   bool                - whether to return the sum of the square aswell
        # PMC            bool                - whether this is run as the first step in the PMC
        # method       t func                - whether to calculate Lyman-alpha or Gaussian test
        # log            bool                - whether to draw from a log distribution
        # returns the summary of the real data, all random draws, the distance between real network output and simulations and the value of the simulated summaries (also real summaries and graphs, sessions and dictionaries)
        #______________________________________________________________ 
        # resetup(W, b)    (session, graph, dictionary)   - creates the network architecture with trained weights and biases applied
        # get_noise(noise, num) ()    - generates noise for Gaussian noise
        # Gaussian(θ, simulation, random_seed)  (data)    - generates the Gaussian noise
        # random_field(θ, simulation, random_seed) (data) - generates the quasar spectrum
        #______________________________________________________________ 
        # session           TF session      - TensorFlow session with initialised network (empty weights and biases)
        # graph             TF graph        - TensorFlow graph with the network architecture
        # dictionary        dict            - dictionary to be fed to the network (includes pretrained weights and biases)
        # summary           float           - network summary of the real data
        # thetas            array           - samples from the prior
        # ind               int             - index counting number of random draws
        # simulated_data    array           - simulation at random draw of theta
        # distance          array           - distance between real summary and simulation      
        # simulated_summary array           - array of network summaries of the simulations
        # r_s               array           - array of the sum of the square of simulations
        session, graph, dictionary = t.n.resetup(W, b)
        dictionary[graph.get_tensor_by_name('x:0')] = real_data
        summary = session.run(t.n.a, feed_dict = dictionary)[-1][0, 0, 0]
        if log:
            thetas = 10.**np.random.uniform(np.log10(prior[0]), np.log10(prior[1]), num_thetas)
        else:
            thetas = np.random.uniform(prior[0], prior[1], num_thetas)
        distance = np.zeros(num_thetas)
        simulated_summary = np.zeros(num_thetas)
        if real_summary:
            r_s = np.zeros(num_thetas)
        if t.method == t.Gaussian:
            t.get_noise(noise, num_thetas)
        for ind in tqdm.tqdm(range(num_thetas)):
            simulated_data = t.method(thetas[ind], ind, shaped = True)
            dictionary[graph.get_tensor_by_name('x:0')] = simulated_data
            simulated_summary[ind] = session.run(t.n.a, feed_dict = dictionary)[-1][0, 0, 0]
            distance[ind] = np.sqrt(F * (summary - simulated_summary[ind])**2.)
            if real_summary:
                r_s[ind] = np.sum(simulated_data**2.)
        if PMC:
            return summary, thetas, simulated_summary, distance, session, graph, dictionary
        elif real_summary:
            return summary, thetas, simulated_summary, distance, r_s
        else:
            return summary, thetas, simulated_summary, distance
    
    def PMC(t, W, b, F, real_data, prior, num_thetas, num_keep, criterion, continues = None, noise = None, log = False):
        # POPULATION MONTE CARLO
        # W              list                - list of all the pretrained weights of the network
        # b              list                - list of all the pretrained biases of the network
        # F              float               - Fisher information for calculating distance
        # real_data      array               - real data to do network comparison with
        # prior          list                - lower and upper bounds of uniform prior
        # num_thetas     int                 - number of random draws from the prior
        # num_thetas     int                 - number of samples in the posterior
        # criterion      float               - ratio of draws to kept samples
        # continues      list                - list of output from PMC to continue at
        # noise          None, list or float - noise for Gaussian model (None use preloaded noise)
        # method       t func                - whether to calculate Lyman-alpha or Gaussian test
        # log            bool                - whether to draw from a log distribution
        # returns the summary, the theta draws, simulated summaries, distances, and sample weights
        #_____________________________________________________________ 
        # ABC(W, b, F, real_data, prior, num_thetas, noise = noise, PMC = True)
        #         (summary, thetas, summaries, rho, session, graph, dictionary)
        #         - initialise graph and session and get random draws from prior
        # resetup(W, b)    (session, graph, dictionary)   - creates the network architecture with trained weights and biases applied
        # get_noise(noise, num) ()    - generates noise for Gaussian noise
        # Gaussian(θ, simulation, random_seed)  (data)    - generates the Gaussian noise
        # random_field(θ, simulation, random_seed) (data) - generates the quasar spectrum
        #______________________________________________________________ 
        # summary           float       - network summary of the real data
        # thetas            array       - samples from the prior
        # rho               array       - distance between real summary and simulation      
        # summaries         array       - array of network summaries of the simulations
        # weighting         array       - weighting of the samples for the covariance
        # session           TF session  - TensorFlow session with initialised network (empty weights and biases)
        # graph             TF graph    - TensorFlow graph with the network architecture
        # dictionary        dict        - dictionary to be fed to the network (includes pretrained weights and biases)
        # prior_at_value    float       - value of the uniform prior in prior region
        # new_thetas        array       - copy of samples from the prior
        # new_weighting     array       - copy of weighting
        # total_draws       int         - counting parameter for total number of draws
        # rounds            int         - counting parameter for number of iterations
        # changes           int         - number of samples moved
        # reach_criterion   float       - ratio of number draws to number of samples
        # cov               array       - weighted covariance of samples
        # epsilon           float       - 75th percentile acceptance
        # i                 int         - index counting number of random draws
        # simulated         array       - simulation at random draw of theta
        #______________________________________________________________                
        if continues is None:
            summary, thetas, summaries, rho, session, graph, dictionary = t.ABC(W, b, F, real_data, prior, num_thetas, noise = noise, PMC = True, log = log)
            index = np.argsort(np.abs(rho))
            thetas = thetas[index[: num_keep]]
            rho = rho[index[: num_keep]]
            summaries = summaries[index[: num_keep]]                        
            weighting = np.ones(num_keep) / num_keep
        else:
            session, graph, dictionary = t.n.resetup(W, b)
            summary = continues[0]
            thetas = continues[1]
            summaries = continues[2]
            rho = continues[3]
            weighting = continues[4]
        prior_at_value = 1./(prior[1] - prior[0])
        new_thetas = np.copy(thetas)
        new_weighting = np.copy(weighting)
        total_draws = num_thetas
        rounds = 0
        changes = -1
        reached_criterion = 1
        while(reached_criterion > criterion):
            if t.method == t.Gaussian:
                t.get_noise(noise, num_thetas)
            draws = 0
            changes = 0
            cov = np.cov(thetas, aweights = weighting)
            epsilon = np.percentile(rho, 75)
            for i in range(num_keep):
                if(rho[i] > epsilon):
                    changes += 1    
                while(rho[i] > epsilon):
                    draws += 1
                    new_thetas[i] = np.random.normal(thetas[i], cov)
                    while((new_thetas[i] <= prior[0]) or (new_thetas[i] >= prior[1])):
                        new_thetas[i] = np.random.normal(thetas[i], np.sqrt(cov))
                    simulated = t.method(new_thetas[i], i, shaped = True)
                    dictionary[graph.get_tensor_by_name('x:0')] = simulated
                    summaries[i] = session.run(t.n.a, feed_dict = dictionary)[-1][0, 0, 0] 
                    rho[i] = np.sqrt(F *(summary - summaries[i])**2.)
                new_weighting[i] = prior_at_value / np.sum(weighting * np.exp(-(new_thetas[i] - thetas)**2. / (2. * cov)) / np.sqrt(2. * np.pi * cov))
            thetas = np.copy(new_thetas)
            weighting = np.copy(new_weighting)
            total_draws += draws
            rounds += 1
            reached_criterion = num_keep / draws
            print(rounds, total_draws, draws, changes, reached_criterion)
        return summary, thetas, summaries, rho, weighting
       
    def asymptotic_likelihood(t, W, b, real_data, data, dtheta, exp = False):
        # CALCULATE THE POSTERIOR FROM GAUSSIAN EXPANSION OF LIKELIHOOD
        # W               list                - list of all the pretrained weights of the network
        # b               list                - list of all the pretrained biases of the network
        # real_data       array               - real data to do network comparison with
        # data            array               - simulations used to train network
        # dtheta          array               - pertubations of theta around fiducial value
        # exp             bool                - whether the dtheta should be exponentiated (for Lyman)
        # training_graph n TF obj             - TensorFlow graph for training the network
        # train_session n TF session          - use training session to get Fisher components
        
        # returns expanded Gaussian between real data and the simulated network data
        #_____________________________________________________________ 
        # resetup(W, b)    (session, graph, dictionary)   - creates the network architecture with trained weights and biases applied
        #______________________________________________________________ 
        # session           TF session  - TensorFlow session with initialised network (empty weights and biases)
        # graph             TF graph    - TensorFlow graph with the network architecture
        # dictionary        dict        - dictionary to be fed to the network (includes pretrained weights and biases)
        # real_a            float       - network summary of the real data
        # dropped           list        - holder for which neurons to drop
        # dictionary_       dict        - dictionary to be fed to the network (includes pretrained weights and biases)
        # invcov_m_dmdt     list        - list of inverse covariance, mean and derivative of the mean
        # invcov            array       - covariance of simulations
        # mean              array       - mean of simulations
        # dmdt              array       - numerical derivatives of the mean of sims
        # fdiC              array       - f(d)^T C^{-1}
        # fdiCfd            array       - f(d)^T C^{-1} f(d)
        # miC               array       - m^T C^{-1}
        # miCfd             array       - m^T C^{-1} f(d)
        # dmdtiC            array       - dm/dt^T C^{-1}
        # dmdtiCfd          array       - dm/dt^T C^{-1} f(d)
        # fdiCm             array       - f(d)^T C^{-1} m
        # fdiCdmdt          array       - f(d)^T C^{-1} dm/dt
        # miCm              array       - m^T C^{-1} m
        # dmdtiCm           array       - dm/dt^T C^{-1} m
        # miCdmdt           array       - m C^{-1} dm/dt
        # dmdtiCdmdt        array       - dm/dt^T C^{-1} dm/dt
        #______________________________________________________________              
        if exp:
            dtheta = np.exp(dtheta)
        session, graph, dictionary = t.n.resetup(W, b)
        dictionary[graph.get_tensor_by_name('x:0')] = real_data
        real_a = session.run(t.n.a, feed_dict = dictionary)[-1]
        dropped = t.n.dropout(d = 0.)
        dictionary_ = t.n.network_dictionary(data, dropped, t.n.training_graph)
        invcov_m_dmdt = t.n.train_session.run(t.n.invcov_m_dmdt, feed_dict = dictionary_)
        invcov = invcov_m_dmdt[0]
        mean = invcov_m_dmdt[1]
        dmdt = invcov_m_dmdt[2]
        fdiC = np.einsum('ijk,ilk->ijl', real_a, invcov)
        fdiCfd = np.einsum('ijk,ilk->ijl', fdiC, real_a)
        miC = np.einsum('ijk,ilk->ijl', mean, invcov)
        miCfd = np.einsum('ijk,ilk->ijl', miC, real_a)
        dmdtiC = np.einsum('ijk,ilk->ijl', dmdt, invcov)
        dmdtiCfd = np.einsum('ijk,ilk->ijl', dmdtiC, real_a) * dtheta
        fdiCm = np.einsum('ijk,ilk->ijl', fdiC, mean)
        fdiCdmdt = np.einsum('ijk,ilk->ijl', fdiC, dmdt) * dtheta
        miCm = np.einsum('ijk,ilk->ijl', miC, mean)
        dmdtiCm = np.einsum('ijk,ilk->ijl', dmdtiC, mean) * dtheta
        miCdmdt = np.einsum('ijk,ilk->ijl', miC, dmdt) * dtheta
        dmdtiCdmdt = np.einsum('ijk,ilk->ijl', dmdtiC, dmdt) * dtheta * dtheta
        return np.einsum('iij->j', ((fdiCfd - miCfd - fdiCm + miCm) - dmdtiCfd - fdiCdmdt + dmdtiCm + miCdmdt + dmdtiCdmdt))
    
    def lnL(t, real_summary, s12, s22, nd):
        # CALCULATE CHI^2 ANALYTICALLY FOR THE GAUSSIAN TEST
        # real_summary       float       - squared sum of the real data
        # s12                float       - value of the real signal variance
        # s22                float       - value of the noise variance
        # nd                 float       - number of inputs
        # returns the chi^2 of the real summary
        #______________________________________________________________              
        return (-0.5 * nd * np.log(2. * np.pi * (s12 + s22)) - 0.5 * real_summary / (s12 + s22))
    
    def lnL_grav(t, real_data, prior, W = None, b = None, MOPED = None):
        # CALCULATE THE LOG LIKELIHOOD OF THE GRAVITATIONAL WAVE CENTRAL FREQUENCY
        # real_data    array            - real data to do network comparison with
        # prior        array            - values of the central values to be evaluated
        # W            list             - the weights for the network
        # b            list             - the biases for the network
        # MOPED        array            - MOPED compression vector
        # returns the log likelihood of the real summary
        #______________________________________________________________ 
        # session     tf session        - initialise the tensorflow session
        # graph       tf graph          - the neural network
        # dictionory     dict           - feed dictionary for the network
        # real           array          - the real data (can be compressed using MOPED or the network)
        # C              float          - normalisation parameter
        # lnL            array          - log likelihood at parameter values
        # ind            int            - frequency counting parameter
        # h_             array          - gravitational waveform in real space
        # waveform       array          - h_ (can be compressed using MOPED or the network)
        # smh_           array          - difference between real data and waveform
        #______________________________________________________________              
        if (W is not None) and (b is not None):
            session, graph, dictionary = t.n.resetup(W, b)
            dictionary[graph.get_tensor_by_name('x:0')] = real_data
            real = session.run(t.n.a, feed_dict = dictionary)[-1][0, 0, 0]
        else:
            real = real_data
        C = 0.
        lnL = np.zeros(prior.shape)
        for ind in range(len(prior)):
            h_ = np.fft.irfft(t.SN * t.gravitational_waveform(prior[ind])).real
            if (W is not None) and (b is not None):
                dictionary[graph.get_tensor_by_name('x:0')] = h_[np.newaxis, np.newaxis, :]
                waveform = session.run(t.n.a, feed_dict = dictionary)[-1][0, 0, 0]
            elif MOPED is None:
                waveform = h_
            else:
                real = np.dot(MOPED, real)
                waveform = np.dot(MOPED, h_)
            smh_ = real - waveform
            lnL[ind] = C - np.dot(smh_, smh_) / 2.
        return lnL
        
class utils():
    def check_error(u, parameter, error, warning, to_break = True):
        # CHECKS ERRORS IN INPUT PARAMETERS
        ind = 0
        err = False
        while (ind < len(error)):
            if type(error[ind]) is type:
                if type(parameter) is not error[ind]:
                    print(warning)
                    parameter = None
                    if to_break:
                        sys.exit()
                    else:
                        err = True
            if error[ind] is 'compare':
                if any([error[ind + 1](parameter, error[ind + 2]), error[ind + 3](parameter, error[ind + 4])]):
                    print(warning)
                    parameter = None
                    if to_break:
                        sys.exit()
                    else:
                        err = True
                ind += 4
            if type(error[ind]) is type(operator.gt):
                if error[ind](parameter, error[ind + 1]):
                    print(warning)
                    parameter = None
                    if to_break:
                        sys.exit()
                    else:
                        err = True
                ind += 1
            ind += 1
        return err
    
    def print_error(u, error, to_break = False):
        # PRINTS ERRORS IF PRESENT
        if error:
            print('There is an error - hope that was helpful')
            if to_break:
                sys.exit()
                
    def ceil(u, val):
        # CEILING VALUE OF INT
        if float(int(val)) < val:
            return int(val) + 1
        else:
            return int(val)

