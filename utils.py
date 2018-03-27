import tensorflow as tf
import numpy as np
import sys

class utils():
    def check_params(u, params):
        # CHECKS PARAMETERS ARE IN THE INITIALISATION DICTIONARY
        #______________________________________________________________
        # CALLED FROM (DEFINED IN IMNN.py)
        # __init__(dict)                          - initialises IMNN
        #______________________________________________________________
        # INPUTS
        # params                        dict      - dictionary containing initialisation parameters
        #   verbose                     bool      - True to print outputs such as shape of tensors
        #   total number of simulations int       - total number of simulations to use
        #   number of parameters        int       - number of parameters in the forward model
        #   number of combinations      int       - number of combinations to split training simulations into
        #   differentiation fraction    int       - fraction of the total number of simulations to use for derivative
        #   prebuild                    bool      - True to allow IMNN to build the network
        #   η                           float     - learning rate
        #   number of summaries         int       - number of outputs from the network
        #   input shape                 int/list  - number of inputs (int) or shape of input (list)
        #______________________________________________________________
        # VARIABLES
        # necessary_parameters          list      - list of necessary parameters
        # key                           str       - key value to check
        #______________________________________________________________
        necessary_parameters = ['verbose', 'total number of simulations', 'number of parameters', 'number of combinations', 'differentiation fraction', 'prebuild', 'η', 'input shape', 'number of summaries']
        for key in necessary_parameters:
            if key not in params.keys():
                print(key + ' not found in parameter dictionary.')
                sys.exit()

    def check_prebuild_params(u, params):
        # CHECKS PARAMETERS IN THE PREBUILD DICTIONARY IF ALLOWING MODULE TO BUILD NETWORK
        #______________________________________________________________
        # CALLED FROM (DEFINED IN IMNN.py)
        # __init__(dict)                          - initialises IMNN
        #______________________________________________________________
        # INPUTS
        # params                        dict      - dictionary containing prebuild parameters
        #   wv                          float     - weight variance to initialise all weights
        #   bb                          float     - constant bias initialiser value
        #   activation                  tf func   - activation function to use
        #   hidden_layers               list      - contains the neural architecture of the network
        #______________________________________________________________
        # VARIABLES
        # prebuild_parameters           list      - list of necessary parameters
        # key                           str       - key value to check
        #______________________________________________________________        
        prebuild_parameters = ['wv', 'bb', 'activation', 'hidden layers']
        for key in prebuild_parameters:
            if key not in params.keys():
                print(key + ' not found in parameter dictionary.')
                sys.exit()
            
    def get_params(u, value, optional):
        # GETS VALUE FROM DICTIONARY OR PASSES VALUE FORWARDS
        #______________________________________________________________
        # CALLED FROM (DEFINED IN utils.py)
        # isboolean(other/list, optional str, optional str)
        #                               bool      - returns boolean if input is boolean
        # isint(other/list, optional str, optional str)
        #                               int       - returns integer if input is integer
        # isfloat(other/list, optional str, optional str)
        #                               float     - returns float if input is float
        # positive_integer(other/list, optional str, optional str)
        #                               int       - returns integer if input is integer and positive
        # constrained_float(other/list, optional str, optional str)
        #                               float     - returns float if input is float beween 0 and 1
        # isint_or_list(other/list, optional str, optional str)
        #                               int/list  - returns integer/list if input is integer/list
        #______________________________________________________________
        # RETURNS
        # other, str
        # unpacked dictionary, error warning text
        #______________________________________________________________
        # INPUTS
        # value                        list/other - list with parameter and key to unpack
        #                                           or a value to pass forward
        # optional                     str/other  - normally a string to use to output error warning
        #______________________________________________________________
        if type(value) is list:
            if type(value[0]) is dict: 
                return value[0][value[1]], value[1]
        return value, optional
    
    def isboolean(u, value, optional = '', key = ''):
        # CHECKS FOR BOOLEAN INPUT
        #______________________________________________________________
        # CALLED FROM (DEFINED IN IMNN.py)
        # __init__(dict)                          - initialises IMNN
        #______________________________________________________________
        # RETURNS
        # bool
        # returns boolean if input is boolean
        #______________________________________________________________
        # INPUTS
        # value                        list/other - list with parameter and key to unpack or 
        #                                           a value to pass forward
        # optional            optional str        - normally a string to use to output error warning
        # key                 optional str        - string to use to indicate key of value
        #______________________________________________________________
        # FUNCTIONS
        # get_params(list/other, str/other)
        #                             other       - unpacks dictionary or passes value forward
        #______________________________________________________________
        value, key = u.get_params(value, key)
        if type(value) != bool:
            print(key + ' must be a boolean. provided type is a ' + str(type(value)) + '. ' + optional)
            sys.exit()
        return value
    
    def isint(u, value, optional = '', key = ''):
        # CHECKS FOR INTEGER INPUT
        #______________________________________________________________
        # CALLED FROM (DEFINED IN utils.py)
        # activation(dict)             tf func, bool, float/int/None
        #                                         - checks that a valid activation function is used
        #______________________________________________________________
        # RETURNS
        # int
        # returns integer if input is integer
        #______________________________________________________________
        # INPUTS
        # value                        list/other - list with parameter and key to unpack or 
        #                                           a value to pass forward
        # optional            optional str        - normally a string to use to output error warning
        # key                 optional str        - string to use to indicate key of value
        #______________________________________________________________
        # FUNCTIONS
        # get_params(list/other, str/other)
        #                             other       - unpacks dictionary or passes value forward
        #______________________________________________________________
        value, key = u.get_params(value, key)
        if type(value) != int:
            print(key + ' must be a integer. provided type is a ' + str(type(value)) + '. ' + optional)
            sys.exit()
        return value
    
    def isfloat(u, value, optional = '', key = ''):
        # CHECKS FOR FLOAT INPUT
        #______________________________________________________________
        # CALLED FROM (DEFINED IN IMNN.py)
        # __init__(dict)                          - initialises IMNN
        #______________________________________________________________
        # RETURNS
        # float
        # returns float if input is float
        #______________________________________________________________
        # INPUTS
        # value                        list/other - list with parameter and key to unpack or 
        #                                           a value to pass forward
        # optional            optional str        - normally a string to use to output error warning
        # key                 optional str        - string to use to indicate key of value
        #______________________________________________________________
        # FUNCTIONS
        # get_params(list/other, str/other)
        #                             other       - unpacks dictionary or passes value forward
        #______________________________________________________________
        value, key = u.get_params(value, key)
        if type(value) != float:
            print(key + ' must be a float. provided type is a ' + str(type(value)) + '. ' + optional)
            sys.exit()
        return value
    
    def positive_integer(u, value, optional = '', key = ''):
        # CHECKS FOR POSITIVE INTEGER
        #______________________________________________________________
        # CALLED FROM (DEFINED IN utils.py)
        # inputs(dict)                 int/list   - returns the input shape as list or size as int
        # hidden_layers(dict, class)   list       - returns the network architecture as a list
        #______________________________________________________________
        # CALLED FROM (DEFINED IN IMNN.py)
        # __init__(dict)                          - initialises IMNN
        #______________________________________________________________
        # RETURNS
        # int
        # returns integer if input is positive integer
        #______________________________________________________________
        # INPUTS
        # value                        list/other - list with parameter and key to unpack or 
        #                                           a value to pass forward
        # optional            optional str        - normally a string to use to output error warning
        # key                 optional str        - string to use to indicate key of value
        #______________________________________________________________
        # FUNCTIONS
        # get_params(list/other, str/other)
        #                             other       - unpacks dictionary or passes value forward
        #______________________________________________________________
        value, key = u.get_params(value, key)
        if type(value) != int:
            print(key + ' must be a positive integer. provided type is a ' + str(type(value)) + '. ' + optional)
            sys.exit()
        if value < 1:
            print(key + ' must be a positive integer. provided value is ' + str(value) + '. ' + optional)
            sys.exit()
        return value
    
    def constrained_float(u, value, optional = '', key = ''):
        # CHECKS FOR FLOAT INPUT BETWEEN 0 AND 1
        #______________________________________________________________
        # CALLED FROM (DEFINED IN utils.py)
        # number_of_derivative_simulations(dict, class)
        #                              int        - the number of sims for numerical derivative
        # activation(dict)             tf func, bool, float/int/None
        #                                         - checks that a valid activation function is used
        #______________________________________________________________
        # CALLED FROM (DEFINED IN IMNN.py)
        # train(list, int, int, float, array, optional list)
        #                              list/list, list
        #                                         - trains the information maximising neural network
        #______________________________________________________________
        # float
        # returns float if input is float between 0 and 1
        #______________________________________________________________
        # INPUTS
        # value                        list/other - list with parameter and key to unpack or 
        #                                           a value to pass forward
        # optional            optional str        - normally a string to use to output error warning
        # key                 optional str        - string to use to indicate key of value
        #______________________________________________________________
        # FUNCTIONS
        # get_params(list/other, str/other)
        #                             other       - unpacks dictionary or passes value forward
        #______________________________________________________________
        value, key = u.get_params(value, key)
        if type(value) != float:
            print(key + ' must be a float between 0 and 1. provided type is a ' + str(type(value)) + '. ' + optional)
            sys.exit()
        if value > 1:
            print(key + ' must be a float between 0 and 1. provided value is ' + str(value) + '. ' + optional)
            sys.exit()
        if value <= 0:
            print(key + ' must be a float between 0 and 1. provided value is ' + str(value) + '. ' + optional)
            sys.exit()
        return value
    
    def isint_or_list(u, value, optional = '', key = ''):
        # CHECKS FOR INTEGER OR LIST
        #______________________________________________________________
        # CALLED FROM (DEFINED IN utils.py)
        # inputs(dict)                 int/list   - returns the input shape as list or size as int
        # hidden_layers(dict, class)   list       - returns the network architecture as a list
        #______________________________________________________________
        # RETURNS
        # int or list
        # returns integer or list if input is an integer or a list
        #______________________________________________________________
        # INPUTS
        # value                        list/other - list with parameter and key to unpack or 
        #                                           a value to pass forward
        # optional            optional str        - normally a string to use to output error warning
        # key                 optional str        - string to use to indicate key of value
        #______________________________________________________________
        # FUNCTIONS
        # get_params(list/other, str/other)
        #                             other       - unpacks dictionary or passes value forward
        #______________________________________________________________
        value, key = u.get_params(value, key)
        if type(value) != int:
            if type(value) != list:
                print(key + ' must be a integer or list. provided type is a ' + str(type(value)) + '. ' + optional)
                sys.exit()
        return value
    
    def number_of_simulations_per_combination(u, n):
        # CALCULATES NUMBER OF SIMULATIONS TO USE DEPENDING ON NUMBER OF COMBINATIONS
        #______________________________________________________________
        # CALLED FROM (DEFINED IN IMNN.py)
        # __init__(dict)                          - initialises IMNN
        #______________________________________________________________
        # RETURNS
        # int
        # returns integer of number of simulations to use depending on number of combinations
        #______________________________________________________________
        # verbose                     n bool      - True to print outputs such as shape of tensors
        # tot_sims                    n int       - total number of simulations to use
        # n_train                     n int       - number of combinations to split training
        #______________________________________________________________
        value = int(n.tot_sims / n.n_train)
        if n.verbose: print('using ' + str(value * n.n_train) + ' simulations out of ' + str(n.tot_sims) + '.')
        return value

    def inputs(u, params):
        # CHECKS SHAPE OR SIZE OF INPUT
        #______________________________________________________________
        # CALLED FROM (DEFINED IN IMNN.py)
        # __init__(dict)                          - initialises IMNN
        #______________________________________________________________
        # RETURNS
        # int or list
        # returns integer or list containing size or shape of input
        #______________________________________________________________
        # INPUTS
        # params                       dict       - dictionary of parameters
        #______________________________________________________________
        # FUNCTIONS (DEFINED IN utils.py)
        # isint_or_list(int/list, optional str)
        #                              int/list   - returns a list or an int if input is list or int
        # positive_integer(int, optional str, optional str)
        #                              int        - returns an integer if input is positive integer
        #______________________________________________________________
        # VARIABLES
        # key                          str        - dictionary key
        #______________________________________________________________
        key = 'input shape'
        value = u.isint_or_list([params, key], optional = 'if using a list the entries must be 3 positive integers.')
        if type(value) == int:
            value = u.positive_integer([params, key])
        else:
            if len(value) != 3:
                print(key + ' must be a list of 3 positive integers. the length of the list is ' + str(len(value)) + '.')
                sys.exit()
            for i in range(3):
                value[i] = u.positive_integer(value[i], optional = 'the problem is at element ' + str(i) + '.', key = key)
        return value

    def number_of_derivative_simulations(u, params, n):
        # CALCULATES NUMBER OF SIMULATIONS TO USE FOR NUMERICAL DERIVATIVE (PER COMBINATION)
        #______________________________________________________________
        # CALLED FROM (DEFINED IN IMNN.py)
        # __init__(dict)                          - initialises IMNN
        #______________________________________________________________
        # RETURNS
        # int
        # returns the number of simulations to use for numerical derivative per combination
        #______________________________________________________________
        # INPUTS
        # params                       dict       - dictionary of parameters
        # n_s                         n int       - number of simulations in each combination
        #______________________________________________________________
        # FUNCTIONS (DEFINED IN utils.py
        # constrained_float(other/list, optional str, optional str)
        #                               float     - returns float if input is float beween 0 and 1
        #______________________________________________________________
        # VARIABLES
        # value                         float     - fraction of simulations to use for derivative
        #______________________________________________________________
        value = u.constrained_float([params, 'differentiation fraction'])
        return int(n.n_s * value)

    def auto_initialise(u, value):
        # CHECKS WHETHER WEIGHTS ARE INITIALISED BY NETWORK OR THROUGH USER INPUT
        #______________________________________________________________
        # CALLED FROM (DEFINED IN IMNN.py)
        # __init__(dict)                          - initialises IMNN
        #______________________________________________________________
        # RETURNS
        # bool
        # True if the value is 0. and False otherwise
        #______________________________________________________________
        # INPUTS
        # value                         float     - input value of the weight variance
        #______________________________________________________________
        if value > 0.:
            return False
        else:
            return True
        
    def activation(u, params):
        # CHECKS IF CHOSEN ACTIVATION FUNCTION IS ALLOWED AND WHETHER EXTRA PARAMETERS ARE NEEDED
        #______________________________________________________________
        # CALLED FROM (DEFINED IN IMNN.py)
        # __init__(dict)                          - initialises IMNN
        #______________________________________________________________
        # RETURNS
        # tf func, bool, float/int
        # returns the activation function, a boolean for whether an extra parameter is needed, and the extra parameter
        #______________________________________________________________
        # INPUTS
        # params                       dict       - dictionary of parameters
        #______________________________________________________________
        # FUNCTIONS (DEFINED IN utils.py
        # constrained_float(other/list, optional str, optional str)
        #                               float     - returns float if input is float beween 0 and 1
        # isint(other/list, optional str, optional str)
        #                               int       - returns integer if input is an integer
        #______________________________________________________________
        # VARIABLES
        # value                         tf func     - 
        #______________________________________________________________
        value = params['activation']
        takes_α = False
        α = None
        if value == tf.nn.relu:
            return value, takes_α, α
        if value == tf.nn.sigmoid:
            return value, takes_α, α
        if value == tf.nn.tanh:
            return value, takes_α, α
        if value == tf.nn.softsign:
            return value, takes_α, α
        if value == tf.nn.softplus:
            return value, takes_α, α
        if value == tf.nn.selu:
            return value, takes_α, α
        if value == tf.nn.relu6:
            return value, takes_α, α
        if value == tf.nn.elu:
            return value, takes_α, α
        if value == tf.nn.crelu:
            return value, takes_α, α
        takes_α = True
        if value == tf.nn.leaky_relu:
            if 'α' not in params.keys():
                print('α is needed to use tf.nn.leaky_relu')
                sys.exit()
            α = u.constrained_float([params, 'α'], optional = 'technically other values are allowed, but it would be strange to use them!')
            return value, takes_α, α
        if value == tf.nn.softmax:
            if 'α' in params.keys():
                α = u.isint([params, 'α'], optional = 'this should be the index of the dimention to sum over.')
            else:
                takes_α = False
            return value, takes_α, α
        if value == tf.nn.log_softmax:
            if 'α' in params.keys():
                α = u.isint([params, 'α'], optional = 'this should be the index of the dimention to sum over.')
            else:
                takes_α = False
            return value, takes_α, α
        print('the requested activation function is not implemented. it probably just needs adding to utils.activation().')
        sys.exit()
        
    def hidden_layers(u, params, n):
        key = 'hidden layers'
        value = params[key]
        if value is None:
            layers = [n.inputs] + [n.n_summaries]
            if n.verbose: print('network architecture is ' + str(layers) + '.')           
            return layers
        if type(value) != list:
            print(key + ' must be a list of hidden layers. provided type is ' + str(type(value)) + '.')
        if len(value) == 0:
            layers = [n.inputs] + [n.n_summaries]
            if n.verbose: print('network architecture is ' + str(layers) + '.')           
            return layers
        hidden_layer = []
        for i in range(len(value)):
            inner_value = u.isint_or_list(value[i], key = 'hidden layers')
            if type(inner_value) == int:
                hidden_layer.append(u.positive_integer(inner_value, key = 'layer ' + str(i + 1) + ' ', optional = 'this value can also be a list.'))
            else:
                if len(inner_value) != 4:
                    print('each convoultional layer in ' + key + ' must be a list of a positive integer (number of filters), two lists which contain two integers (x and y kernal size in the first list and x and y strides in the second) and finally a string of either "SAME" or "VALID" for padding type). the length of the list is ' + str(len(inner_value)) + '. an integer value can also be used.')
                    sys.exit()
                for j in range(4):
                    if j == 0: 
                        inner_value[j] = u.positive_integer(inner_value[j], optional = 'the problem is at element ' + str(i) + ' which should be an integer.', key = key)
                    elif (j == 1) or (j == 2):
                        if type(inner_value[j]) != list:
                            print('element ' + str(j) + ' of hidden layer ' + str(i + 1) + ' must be a list. provided type is ' + str(type(inner_value[j])) + '.')
                            sys.exit()
                        if len(inner_value[j]) != 2:
                            if j == 1:
                                print('element 1 of hidden layer ' + str(i + 1) + ' list must be a list with two positive integers which describe the shape of the x and y kernel in the convolution. the provided length is ' + str(len(inner_value[j])) + '.')
                                sys.exit()
                            if j == 2:
                                print('element 2 of hidden layer ' + str(i + 1) + 'list must be a list with two positive integers which describe the strides in the x and y direction in the convolution. the provided length is ' + str(len(inner_value[j])) + '.')
                                sys.exit() 
                        for k in range(2): 
                            inner_value[j][k] = u.positive_integer(inner_value[j][k], optional = 'the problem is at element ' + str(k) + ' of element ' + str(j) + ' of hidden layer ' + str(i + 1) + '.', key = 'hidden layer')
                    else:
                        if type(inner_value[j]) != str:
                            print('element ' + str(j) + ' of hidden layer ' + str(i + 1) + ' must be a string of either "SAME" or "VALID" for padding type. provided type ' + str(type(inner_value[j])) + '.')
                            sys.exit()
                        if (inner_value[j] != 'SAME') and (inner_value[j] != 'VALID'):
                            print('element ' + str(j) + ' of hidden layer ' + str(i + 1) + ' must be a string of either "SAME" or "VALID" for padding type. provided string ' + inner_value[j] + '.')
                            sys.exit()
                hidden_layer.append(inner_value)
        layers = [n.inputs] + hidden_layer + [n.n_summaries]
        if n.verbose: print('network architecture is ' + str(layers) + '.')
        return layers
            
    def initialise_variables(u):
        return [None for i in range(11)]

    def to_prebuild(u, network):
        if network is None:
            print('network architecture needs to be prebuilt')
            sys.exit()
            
    def to_test(u, test_data):
        if test_data is None:
            return False
        else:
            if type(test_data) is list:
                if len(test_data) == 3:
                    for i in range(3):
                        if type(test_data[i]) == np.ndarray:
                            return True
            print('test data is provided, but of the wrong form.')
            sys.exit()
        return False
        
    def to_continue(u, samples):
        if samples is None:
            return False
        else:
            if type(samples) == list:
                if len(samples) == 6:
                    if type(samples[0]) == np.ndarray:
                        if type(samples[1]) == nd.ndarray:
                            if type(samples[2]) == np.ndarray:
                                if type(samples[3]) == np.ndarray:
                                    if type(samples[4]) == np.ndarray:
                                        u.postive_integer(samples[5], key = 'element 5 of samples', optional = 'this should be the total number of draws so far.')
                                        return True
                                    print('element 4 of samples should be an array of the sample weights. current type is ' + str(type(samples[4])) + '.')
                                    sys.exit()
                                print('element 3 of samples should be an array of the summaries of the current simulations. current type is ' + str(type(samples[3])) + '.')
                                sys.exit()
                            print('element 2 of samples should be an array of the current distances between simulation summaries and real summary. current type is ' + str(type(samples[2])) + '.')
                            sys.exit()
                        print('element 1 of samples should be an array of the summary of the real data. current type is ' + str(type(samples[1])) + '.')
                        sys.exit()
                    print('element 0 of samples should be an array of the current drawn parameter samples. current type is ' + str(type(samples[0])) + '.')
                    sys.exit()
            print('samples should be a list containing current parameter samples, summary of real data, distances between current summaries and summary of real data, current summaries of simulations, weights for each sample and the total number of draws so far.')
            sys.exit()

    def enough(u, high, low, modulus = False, tight = False):
        if modulus: 
            key_1 = 'num_batches'
            key_2 = 'n_train'
        else:
            key_1 = 'num_keep'
            key_2 = 'num_draws'
        if tight:
            if high <= low:
                print(key_1 + ' must be less than ' + key_2 + '.')
                sys.exit()
        else:
            if high < low:
                print(key_1 + ' must be less than ' + key_2 + '.')
                sys.exit()
        if modulus:
            if high%low != 0:
                print('number of combinations needs to be divisible by the number of batches.')
                sys.exit()