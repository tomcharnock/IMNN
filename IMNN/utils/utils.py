"""A bunch of utility functions for checking stuff in the IMNN

A lot of the functions are complete and working but the network builder needs
to be checked.

## TODO: Tidy up comments in functions for the network building function
"""


__version__ = '0.1dev5'
__author__ = "Tom Charnock"


import tensorflow as tf
import numpy as np
import sys


def load_parameters(parameters):
    """Checks members of the IMNN input parameters dictionary are at least of
    correct type and follow the correct rules.

    Parameters
    __________
    parameters : dict
        dictionary containing all the necessary input parameters for setting up
        the information maxmising neural network.
    loaded_parameters : dict
        dictionary containing all the loaded parameters for the IMNN
        "_FLOATX" : :obj:`TF type`
            32 bit or 64 TensorFlow tensor floats.
        "_INTX" : :obj:`TF type`
            32 bit or 64 TensorFlow tensor integers.
        "n_s" : int
            number of simulations to calculate summary covariance.
        "n_p" : int
            number of derivatives simulations to calculate derivative of mean.
        "n_summaries" : int
            number of summaries to compress data to.
        "n_params" : int
            number of parameters in Fisher information matrix.
        "fiducial" : :obj:`list` of :obj:`float`
            fiducial parameter values to train IMNN at.
        "input_shape" : :obj:`list` of :obj:`int`
            shape of the input data.
        "filename" : str
            filename for saving and loading network.
    Returns
    _______
    dict
        returns the loaded_parameters dictionary
    """
    loaded_parameters = {}
    if "dtype" in parameters.keys():
        if type(parameters["dtype"]) != int:
            print("dtype must be either 32 or 64")
            sys.exit()
        if (parameters["dtype"] != int(32) and parameters["dtype"] != int(64)):
            print("dtype must be 32 or 64. default value is 32")
            sys.exit()
        if parameters["dtype"] == int(32):
            loaded_parameters["_FLOATX"] = tf.float32
            loaded_parameters["_INTX"] = tf.int32
        if parameters["dtype"] == int(64):
            loaded_parameters["_FLOATX"] = tf.float64
            loaded_parameters["_INTX"] = tf.int64
    else:
        _FLOATX = tf.float32
        _INTX = tf.int32

    if "number of simulations" in parameters.keys():
        if type(parameters["number of simulations"]) != int:
            print("number of simulations to calculate covariance must be an \
            integer but is " + str(type(parameters["number of simulations"])))
            sys.exit()
        if parameters["number of simulations"] <= 0:
            print("number of simulations to calculate covariance must be \
            positive but is " + str(parameters["number of simulations"]))
            sys.exit()
        loaded_parameters["n_s"] = parameters["number of simulations"]
    else:
        print("number of simulations to calculate covariance must be provided")
        sys.exit()

    if "number of derivative simulations" in parameters.keys():
        if type(parameters["number of derivative simulations"]) != int:
            print("number of simulations to calculate derivative of the mean \
            must be an integer but is "
                  + str(type(parameters["number of derivative simulations"])))
        if parameters["number of derivative simulations"] <= 0:
            print("number of simulations to calculate derivative of the mean \
            must be positive but is "
                  + str(parameters["number of derivative simulations"]))
        key = "number of derivative simulations"
        loaded_parameters["n_p"] = parameters[key]
    else:
        print("number of simulations to calculate derivative of the mean must \
        be provided")
        sys.exit()

    if "number of summaries" in parameters.keys():
        if type(parameters["number of summaries"]) != int:
            print("number of summaries to compress the data to must be an \
            integer but is " + str(type(parameters["number of summaries"])))
        if parameters["number of summaries"] <= 0:
            print("number of summaries to compress the data to must be \
            positive but is " + str(parameters["number of summaries"]))
        loaded_parameters["n_summaries"] = parameters["number of summaries"]
    else:
        print("number of summaries to compress the data to must be provided")
        sys.exit()

    if "fiducial" in parameters.keys():
        if type(parameters["fiducial"]) != list:
            print("fiducial parameters (fiducial) must be a list but is "
                  + str(type(parameters["fiducial"])))
            sys.exit()
        if len(np.array(parameters["fiducial"]).shape) > 1:
            print("fiducial parameters (fiducial) must be a 1D list but has "
                  + str(len(np.array(parameters["fiducial"]))))
            sys.exit()
        if len(parameters["fiducial"]) == 0:
            print("fiducial parameters (fiducial) must have at least one \
            parameter")
            sys.exit()
        loaded_parameters["fiducial"] = parameters["fiducial"]
        loaded_parameters["n_params"] = len(parameters["fiducial"])
    else:
        print("array of fiducial parameters (fiducial) must be provided")
        sys.exit()

    if "input shape" in parameters.keys():
        if type(parameters["input shape"]) != list:
            print("input shape must be a list of positive integers but is "
                  + str(type(parameters["input shape"])))
            sys.exit()
        inputs = [type(parameters["input shape"][i]) != int
                  for i in range(len(parameters["input shape"]))]
        if any(inputs):
            print("input shape must be a list of positive integers but "
                  + str(np.argwhere(inputs)) + " does not conform")
            sys.exit()
        inputs = [parameters["input shape"][i] <= 0
                  for i in range(len(parameters["input shape"]))]
        if any(inputs):
            print("input shape must be a list of positive integers but "
                  + str(np.argwhere(inputs)) + " does not conform")
            sys.exit()
        loaded_parameters["input_shape"] = parameters["input shape"]
    else:
        print("input shape must be provided")
        sys.exit()

    if "filename" in parameters.keys():
        if type(parameters["filename"]) != str:
            print("filename must be a string but is "
                  + str(type(parameters["filename"])))
            sys.exit()
        loaded_parameters["filename"] = parameters["filename"]
    else:
        loaded_parameters["filename"] = None

    return loaded_parameters


def check_data(n, data):
    """Check the size and shape of the data

    Parameters
    __________
    n : :obj:`class`
        IMNN class object containing the number of necessary simulations for
        calculating the Fisher information and the shape of the input data
    data : dict
        dictionary containing the training and validation data in "data" and
        the training and validation derivatives in "data_d"
    """
    if type(data) != dict:
        print("data must be a dictionary with numpy arrays for training_data \
        and training_data_d")
        sys.exit()
    if "data" not in data.keys():
        print("data must contain, data and data_d")
        sys.exit()
    if "data_d" not in data.keys():
        print("data must contain, data and data_d")
        sys.exit()
    if type(data["data"]) != np.ndarray:
        print("data must be a numpy array")
        sys.exit()
    if data["data"].shape[0] < n.n_s:
        print("there needs to be enough training data to compute the \
        covariance")
        sys.exit()
    if data["data_d"].shape[0] < n.n_p:
        print("there needs to be enough derivatives of the data to compute \
        the mean of the derivative")
        sys.exit()
    if list(data["data"].shape[1:]) != n.input_shape:
        print("data must have the same shape as the defined input shape")
        sys.exit()
    if list(data["data_d"].shape[1:]) != [n.n_params] + n.input_shape:
        print("data_d must have the shape [n_params] + input_shape")
        sys.exit()


def check_amounts(num_sims, num_partial_sims, num_validation_sims,
                  num_validation_partial_sims, data_size, data_d_size):
    """
    """
    num_sims = positive_integer(num_sims, key="num_sims")
    num_partial_sims = positive_integer(num_partial_sims,
                                        key="num_partial_sims")
    num_validation_sims = isint(num_validation_sims,
                                key="num_validation_sims")
    num_validation_partial_sims = isint(num_validation_partial_sims,
                                        key="num_validation_sims")
    if num_validation_sims < 0:
        print("num_validation_sims should be greater than or equal to zero")
        sys.exit()
    if num_validation_partial_sims < 0:
        print("num_validation_partial_sims should be greater than or equal to \
        zero")
        sys.exit()
    if num_sims + num_validation_sims != data_size:
        print("the number of supplied simulations and validation simulations \
        is " + str(num_sims + num_validation_sims) + " but the data is of \
        size " + str(data_size))
        sys.exit()
    if num_partial_sims + num_validation_partial_sims != data_d_size:
        print("the number of supplied partial simulations and validation \
        partial simulations is " + str(num_partial_sims
                                       + num_validation_partial_sims)
                                 + " but the derivative data (data_d) is of \
                                 size " + str(data_d_size))
        sys.exit()
    if num_validation_sims == 0:
        test = False
    else:
        test = True
    return test


def get_params(value, optional):
    """
    """
    if type(value) is list:
        if type(value[0]) is dict:
            return value[0][value[1]], value[1]
    return value, optional


def isint(value, optional='', key=''):
    """
    """
    value, key = get_params(value, key)
    if type(value) != int:
        print(key + ' must be a integer. provided type is a '
              + str(type(value)) + '. ' + optional)
        sys.exit()
    return value


def isfloat(value, optional='', key=''):
    """
    """
    value, key = get_params(value, key)
    if type(value) != float:
        print(key + ' must be a float. provided type is a '
              + str(type(value)) + '. ' + optional)
        sys.exit()
    return value


def positive_integer(value, optional='', key=''):
    """
    """
    value, key = get_params(value, key)
    if type(value) != int:
        print(key + ' must be a positive integer. provided type is a '
              + str(type(value)) + '. ' + optional)
        sys.exit()
    if value < 1:
        print(key + ' must be a positive integer. provided value is '
              + str(value) + '. ' + optional)
        sys.exit()
    return value


def constrained_float(value, optional='', key=''):
    """
    """
    value, key = get_params(value, key)
    if type(value) != float:
        print(key + ' must be a float between 0 and 1. provided type is a '
              + str(type(value)) + '. ' + optional)
        sys.exit()
    if value > 1:
        print(key + ' must be a float between 0 and 1. provided value is '
              + str(value) + '. ' + optional)
        sys.exit()
    if value <= 0:
        print(key + ' must be a float between 0 and 1. provided value is '
              + str(value) + '. ' + optional)
        sys.exit()
    return value


def isnotebook():
    """
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False

# def islist(u, value, optional = "", key = ""):
#     # CHECKS FOR LIST
#     #______________________________________________________________
#     # CALLED FROM (DEFINED IN utils.py)
#     # inputs(dict)                 list       - returns the input shape
#     #______________________________________________________________
#     # RETURNS
#     # list
#     # returns list if input is a list
#     #______________________________________________________________
#     # INPUTS
#     # value                        list/other - list with parameter and key to unpack or
#     #                                           a value to pass forward
#     # optional            optional str        - normally a string to use to output error warning
#     # key                 optional str        - string to use to indicate key of value
#     #______________________________________________________________
#     # FUNCTIONS
#     # get_params(list/other, str/other)
#     #                             other       - unpacks dictionary or passes value forward
#     #______________________________________________________________
#     value, key = u.get_params(value, key)
#     if type(value) != list:
#         print(key + ' must be a list. provided type is a ' + str(type(value)) + '. ' + optional)
#         sys.exit()
#     return value
#
# def isint_or_list(u, value, optional = '', key = ''):
#     # CHECKS FOR INTEGER OR LIST
#     #______________________________________________________________
#     # CALLED FROM (DEFINED IN utils.py)
#     # hidden_layers(dict, class)   list       - returns the network architecture as a list
#     #______________________________________________________________
#     # RETURNS
#     # int or list
#     # returns integer or list if input is an integer or a list
#     #______________________________________________________________
#     # INPUTS
#     # value                        list/other - list with parameter and key to unpack or
#     #                                           a value to pass forward
#     # optional            optional str        - normally a string to use to output error warning
#     # key                 optional str        - string to use to indicate key of value
#     #______________________________________________________________
#     # FUNCTIONS
#     # get_params(list/other, str/other)
#     #                             other       - unpacks dictionary or passes value forward
#     #______________________________________________________________
#     value, key = u.get_params(value, key)
#     if type(value) != int:
#         if type(value) != list:
#             print(key + ' must be a integer or list. provided type is a ' + str(type(value)) + '. ' + optional)
#             sys.exit()
#     return value
#
# def positive_divisible(u, params, key, check, check_key):
#     # CHECKS THAT INPUT IS INTEGER DIVISIBLE AND POSITIVE DEFINITE
#     #______________________________________________________________
#     # CALLED FROM (DEFINED IN IMNN.py)
#     # __init__(dict)                          - initialises IMNN
#     #______________________________________________________________
#     # RETURNS
#     # int
#     # returns integer for the batch size or the batch size for derivatives
#     #______________________________________________________________
#     # INPUTS
#     # params                       dict       - dictionary of parameters
#     # key                          str        - denominator value
#     # check                        int        - numerator value
#     # check_key                    str        - numerator key
#     #______________________________________________________________
#     # FUNCTIONS (DEFINED IN utils.py)
#     # positive_integer(int, optional str, optional str)
#     #                              int        - returns an integer if input is positive integer
#     #______________________________________________________________
#     # VARIABLES
#     #______________________________________________________________
#     if float(check // params[key]) != check / params[key]:
#         print(check_key + " / " + key + " is not an integer")
#         sys.exit()
#     else:
#         u.positive_integer(check // params[key], key = check_key + " / " + key)
#     return check // params[key]
#
#
# def activation(u, params):
#     # CHECKS IF CHOSEN ACTIVATION FUNCTION IS ALLOWED AND WHETHER EXTRA PARAMETERS ARE NEEDED
#     #______________________________________________________________
#     # CALLED FROM (DEFINED IN IMNN.py)
#     # __init__(dict)                          - initialises IMNN
#     #______________________________________________________________
#     # RETURNS
#     # tf func, bool, float/int
#     # returns the activation function, a boolean for whether an extra parameter is needed, and the extra parameter
#     #______________________________________________________________
#     # INPUTS
#     # params                        dict      - dictionary of parameters
#     #______________________________________________________________
#     # FUNCTIONS (DEFINED IN utils.py)
#     # constrained_float(other/list, optional str, optional str)
#     #                               float     - returns float if input is float beween 0 and 1
#     # isint(other/list, optional str, optional str)
#     #                               int       - returns integer if input is an integer
#     #______________________________________________________________
#     # VARIABLES
#     # value                         tf func   - tensorflow activation function
#     # takes_α                       bool      - True if activation function needs an input parameter
#     # α                             float/int - value of the parameter needed for the activation function
#     #______________________________________________________________
#     value = params['activation']
#     takes_α = False
#     α = None
#     if value == tf.nn.relu:
#         return value, takes_α, α
#     if value == tf.nn.sigmoid:
#         return value, takes_α, α
#     if value == tf.nn.tanh:
#         return value, takes_α, α
#     if value == tf.nn.softsign:
#         return value, takes_α, α
#     if value == tf.nn.softplus:
#         return value, takes_α, α
#     if value == tf.nn.selu:
#         return value, takes_α, α
#     if value == tf.nn.relu6:
#         return value, takes_α, α
#     if value == tf.nn.elu:
#         return value, takes_α, α
#     if value == tf.nn.crelu:
#         return value, takes_α, α
#     takes_α = True
#     if value == tf.nn.leaky_relu:
#         if 'α' not in params.keys():
#             print('α is needed to use tf.nn.leaky_relu')
#             sys.exit()
#         α = u.constrained_float([params, 'α'], optional = 'technically other values are allowed, but it would be strange to use them!')
#         return value, takes_α, α
#     if value == tf.nn.softmax:
#         if 'α' in params.keys():
#             α = u.isint([params, 'α'], optional = 'this should be the index of the dimention to sum over.')
#         else:
#             takes_α = False
#         return value, takes_α, α
#     if value == tf.nn.log_softmax:
#         if 'α' in params.keys():
#             α = u.isint([params, 'α'], optional = 'this should be the index of the dimention to sum over.')
#         else:
#             takes_α = False
#         return value, takes_α, α
#     print('the requested activation function is not implemented. it probably just needs adding to utils.activation().')
#     sys.exit()
#
# def hidden_layers(u, params, n):
#     # CHECKS ARCHITECTURE OF PREBUILT NEURAL NETWORK
#     #______________________________________________________________
#     # CALLED FROM (DEFINED IN IMNN.py)
#     # __init__(dict)                          - initialises IMNN
#     #______________________________________________________________
#     # RETURNS
#     # list
#     # neural network architecture for the IMNN
#     #______________________________________________________________
#     # INPUTS
#     # params                        dict      - dictionary of parameters
#     # inputs                      n int/list  - number of inputs (int) or shape of input (list)
#     # n_summaries                 n int       - number of outputs from the network
#     # verbose                     n bool      - True to print outputs such as shape of tensors
#     #______________________________________________________________
#     # FUNCTIONS (DEFINED IN utils.py)
#     # isint(other/list, optional str, optional str)
#     #                               int       - returns integer if input is an integer
#     # isint_or_list(int/list, optional str)
#     #                              int/list   - returns a list or an int if input is list or int
#     # positive_integer(int, optional str, optional str)
#     #                              int        - returns an integer if input is positive integer
#     #______________________________________________________________
#     # VARIABLES
#     # key                           str       - dictionary key
#     # value                         float     - fraction of simulations to use for derivative
#     # layers                        list      - list of network architecture
#     # hidden_layer                  list      - list of hidden layers in neural network
#     # i                             int       - counter for layers
#     # inner_value                   int/list  - description of hidden layer (dense or convolutional)
#     # j                             int       - counter for convolutional layer parameters
#     # k                             int       - counter for convolutional kernel size or stride size
#     #______________________________________________________________
#     key = 'hidden layers'
#     value = params[key]
#     if value is None:
#         layers = [n.inputs] + [n.n_summaries]
#         if n.verbose: print('network architecture is ' + str(layers) + '.')
#         return layers
#     if type(value) != list:
#         print(key + ' must be a list of hidden layers. provided type is ' + str(type(value)) + '.')
#     if len(value) == 0:
#         layers = [n.inputs] + [n.n_summaries]
#         if n.verbose: print('network architecture is ' + str(layers) + '.')
#         return layers
#     hidden_layer = []
#     for i in range(len(value)):
#         inner_value = u.isint_or_list(value[i], key = 'hidden layers')
#         if type(inner_value) == int:
#             hidden_layer.append(u.positive_integer(inner_value, key = 'layer ' + str(i + 1) + ' ', optional = 'this value can also be a list.'))
#         else:
#             if len(inner_value) != 4:
#                 print('each convoultional layer in ' + key + ' must be a list of a positive integer (number of filters), two lists which contain two integers (kernel sizes in the first list and strides in the second) and finally a string of either "SAME" or "VALID" for padding type). the length of the list is ' + str(len(inner_value)) + '. an integer value can also be used.')
#                 sys.exit()
#             for j in range(4):
#                 if j == 0:
#                     inner_value[j] = u.positive_integer(inner_value[j], optional = 'the problem is at element ' + str(i) + ' which should be an integer.', key = key)
#                 elif (j == 1) or (j == 2):
#                     if type(inner_value[j]) != list:
#                         print('element ' + str(j) + ' of hidden layer ' + str(i + 1) + ' must be a list. provided type is ' + str(type(inner_value[j])) + '.')
#                         sys.exit()
#                     if len(inner_value[j]) < 1 or len(inner_value[j]) > 3:
#                         if j == 1:
#                             print('element 1 of hidden layer ' + str(i + 1) + ' list must be a list with one, two or three positive integers for 1D, 2D or 3D convolutions which describe the shape of the kernel in the convolution. the provided length is ' + str(len(inner_value[j])) + '.')
#                             sys.exit()
#                         if j == 2:
#                             print('element 2 of hidden layer ' + str(i + 1) + 'list must be a list with one, two or three positive integers for 1D, 2D or 3D convolutions which describe the strides in the convolution. the provided length is ' + str(len(inner_value[j])) + '.')
#                             sys.exit()
#                     for k in range(len(inner_value[j])):
#                         inner_value[j][k] = u.positive_integer(inner_value[j][k], optional = 'the problem is at element ' + str(k) + ' of element ' + str(j) + ' of hidden layer ' + str(i + 1) + '.', key = 'hidden layer')
#                 else:
#                     if type(inner_value[j]) != str:
#                         print('element ' + str(j) + ' of hidden layer ' + str(i + 1) + ' must be a string of either "SAME" or "VALID" for padding type. provided type ' + str(type(inner_value[j])) + '.')
#                         sys.exit()
#                     if (inner_value[j] != 'SAME') and (inner_value[j] != 'VALID'):
#                         print('element ' + str(j) + ' of hidden layer ' + str(i + 1) + ' must be a string of either "SAME" or "VALID" for padding type. provided string ' + inner_value[j] + '.')
#                         sys.exit()
#             hidden_layer.append(inner_value)
#     layers = [n.inputs] + hidden_layer + [n.n_summaries]
#     if n.verbose: print('network architecture is ' + str(layers) + '.')
#     return layers
#
# def initialise_variables(u):
#     # INITIALISES ALL SHARED NETWORK PARAMETERS TO NONE
#     #______________________________________________________________
#     # CALLED FROM (DEFINED IN IMNN.py)
#     # __init__(dict)                          - initialises IMNN
#     #______________________________________________________________
#     # RETURNS
#     # list
#     # returns a list of None for each unset shared network parameter
#     #______________________________________________________________
#     return [None for i in range(29)]
#
# def to_prebuild(u, network):
#     # INITIALISES ALL SHARED NETWORK PARAMETERS TO NONE
#     #______________________________________________________________
#     # CALLED FROM (DEFINED IN IMNN.py)
#     # setup(optional func)                    - sets up the network and initialises it
#     #______________________________________________________________
#     # INPUTS
#     # network                       func      - function for hidden layers of the network
#     #______________________________________________________________
#     if network is None:
#         print('network architecture needs to be prebuilt')
#         sys.exit()
#
# def to_continue(u, samples):
#     # CHECKS LIST OF NECESSARY PMC COMPONENTS TO SEE WHETHER PMC CAN CONTINUE
#     #______________________________________________________________
#     # CALLED FROM (DEFINED IN IMNN.py)
#     # PMC(array, array, list, int, int, func, float, optional bool, optional list)
#     #                               array, float, array, array, array, int
#     #                                         - performs PMC-ABC using the IMNN
#     #______________________________________________________________
#     # INPUTS
#     # samples                       list      - list where the components are sampled parameter values, network summary
#     #                                           of real data, distances between simulation summaries and real summary,
#     #                                           summaries of simulations, weighting of samples, total number of draws
#     #______________________________________________________________
#     # FUNCTIONS (DEFINED IN utils.py)
#     # positive_integer(int, optional str, optional str)
#     #                              int        - returns an integer if input is positive integer
#     #______________________________________________________________
#     if samples is None:
#         return False
#     else:
#         if type(samples) == list:
#             if len(samples) == 7:
#                 if type(samples[0]) == np.ndarray:
#                     if type(samples[1]) == np.ndarray:
#                         if type(samples[2]) == np.ndarray:
#                             if type(samples[3]) == np.ndarray:
#                                 if type(samples[4]) == np.ndarray:
#                                     u.positive_integer(samples[5], key = 'element 5 of samples', optional = 'this should be the total number of draws so far.')
#                                     if type(samples[6]) == np.ndarray:
#                                         return True
#                                     print('element 6 of samples should be an array of the Fisher information matrix. current type is ' + str(type(samples[6])) + '.')
#                                 print('element 4 of samples should be an array of the sample weights. current type is ' + str(type(samples[4])) + '.')
#                                 sys.exit()
#                             print('element 3 of samples should be an array of the summaries of the current simulations. current type is ' + str(type(samples[3])) + '.')
#                             sys.exit()
#                         print('element 2 of samples should be an array of the current distances between simulation summaries and real summary. current type is ' + str(type(samples[2])) + '.')
#                         sys.exit()
#                     print('element 1 of samples should be an array of the summary of the real data. current type is ' + str(type(samples[1])) + '.')
#                     sys.exit()
#                 print('element 0 of samples should be an array of the current drawn parameter samples. current type is ' + str(type(samples[0])) + '.')
#                 sys.exit()
#         print('samples should be a list containing current parameter samples, summary of real data, distances between current summaries and summary of real data, current summaries of simulations, weights for each sample and the total number of draws so far.')
#         sys.exit()
#
# def enough(u, high, low, modulus = False, tight = False):
#     # CHECKS THAT ONE NUMBER IS HIGHER THAN ANOTHER
#     #______________________________________________________________
#     # CALLED FROM (DEFINED IN IMNN.py)
#     # train(list, int, int, float, array, optional list)
#     #                              list/list, list
#     #                                         - trains the information maximising neural network
#     # PMC(array, array, list, int, int, func, float, optional bool, optional list)
#     #                               array, float, array, array, array, int
#     #                                         - performs PMC-ABC using the IMNN
#     #______________________________________________________________
#     # INPUTS
#     # high                          int       - value which should be higher
#     # low                           int       - value which should be lower
#     # modulus                       bool      - True if higher value must be multiple of lower value
#     # tight                         bool      - True if lower value must be smaller and not equal to higher value
#     #______________________________________________________________
#     # VARIABLES
#     # key_1                         str       - name of lower valued entry
#     # key_2                         str       - name of higher valued entry
#     #______________________________________________________________
#     if modulus:
#         key_1 = 'num_batches'
#         key_2 = 'n_train'
#     else:
#         key_1 = 'num_keep'
#         key_2 = 'num_draws'
#     if tight:
#         if high <= low:
#             print(key_1 + ' must be less than ' + key_2 + '.')
#             sys.exit()
#     else:
#         if high < low:
#             print(key_1 + ' must be less than ' + key_2 + '.')
#             sys.exit()
#     if modulus:
#         if high%low != 0:
#             print('number of combinations needs to be divisible by the number of batches.')
#             sys.exit()
#
# def check_params(u, params):
#     # CHECKS PARAMETERS ARE IN THE INITIALISATION DICTIONARY
#     #______________________________________________________________
#     # CALLED FROM (DEFINED IN IMNN.py)
#     # __init__(dict)                          - initialises IMNN
#     #______________________________________________________________
#     # INPUTS
#     # params                        dict      - dictionary containing initialisation parameters
#     #   verbose                     bool      - True to print outputs such as shape of tensors
#     #   number of simulations       int       - number of simulations to use per combination
#     #   number of parameters        int       - number of parameters in the forward model
#     #   differentiation fraction    int       - fraction of the total number of simulations to use for derivative
#     #   prebuild                    bool      - True to allow IMNN to build the network
#     #   number of summaries         int       - number of outputs from the network
#     #   input shape                 list      - number of inputs (int) or shape of input (list)
#     #   calculate MLE               bool      - True to calculate the maximum likelihood estimate
#     #   preload data                dict/None - data to preload as a TensorFlow constant
#     #   save file                   str/None  - path and file name to save (or load) data
#     #______________________________________________________________
#     # VARIABLES
#     # necessary_parameters          list      - list of necessary parameters
#     # key                           str       - key value to check
#     #______________________________________________________________
#     necessary_parameters = ['verbose', 'number of simulations', 'fiducial θ', 'derivative denominator', 'differentiation fraction', 'prebuild', 'input shape', 'number of summaries', 'calculate MLE', 'preload data', 'save file']
#     for key in necessary_parameters:
#         if key not in params.keys():
#             print(key + ' not found in parameter dictionary.')
#             sys.exit()
#
# def check_prebuild_params(u, params):
#     # CHECKS PARAMETERS IN THE PREBUILD DICTIONARY IF ALLOWING MODULE TO BUILD NETWORK
#     #______________________________________________________________
#     # CALLED FROM (DEFINED IN IMNN.py)
#     # __init__(dict)                          - initialises IMNN
#     #______________________________________________________________
#     # INPUTS
#     # params                        dict      - dictionary containing prebuild parameters
#     #   wv                          float     - weight variance to initialise all weights
#     #   bb                          float     - constant bias initialiser value
#     #   activation                  tf func   - activation function to use
#     #   hidden_layers               list      - contains the neural architecture of the network
#     #______________________________________________________________
#     # VARIABLES
#     # prebuild_parameters           list      - list of necessary parameters
#     # key                           str       - key value to check
#     #______________________________________________________________
#     prebuild_parameters = ['wv', 'bb', 'activation', 'hidden layers']
#     for key in prebuild_parameters:
#         if key not in params.keys():
#             print(key + ' not found in parameter dictionary.')
#             sys.exit()
#
# def isboolean(u, value, optional = '', key = ''):
#     # CHECKS FOR BOOLEAN INPUT
#     #______________________________________________________________
#     # CALLED FROM (DEFINED IN IMNN.py)
#     # __init__(dict)                          - initialises IMNN
#     #______________________________________________________________
#     # RETURNS
#     # bool
#     # returns boolean if input is boolean
#     #______________________________________________________________
#     # INPUTS
#     # value                        list/other - list with parameter and key to unpack or
#     #                                           a value to pass forward
#     # optional            optional str        - normally a string to use to output error warning
#     # key                 optional str        - string to use to indicate key of value
#     #______________________________________________________________
#     # FUNCTIONS
#     # get_params(list/other, str/other)
#     #                             other       - unpacks dictionary or passes value forward
#     #______________________________________________________________
#     value, key = u.get_params(value, key)
#     if type(value) != bool:
#         print(key + ' must be a boolean. provided type is a ' + str(type(value)) + '. ' + optional)
#         sys.exit()
#     return value
#
