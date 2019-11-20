import sys
import numpy as np

verbose = True


def positive_integer(param, name):
    if type(param) != int:
        error = True
    elif param < 1:
        error = True
    else:
        error = False
    if error:
        if verbose:
            print(name + " must be a positive integer but has a value of "
                  + str(param))
        sys.exit()
    else:
        return param


def bool_none(param):
    if param is not None:
        a = True
    else:
        a = False
    return a


def type_warning(type, wanted, variable, message=None):
    if type is None:
        if verbose:
            print(variable + " cannot be None")
            sys.exit()
    if type != wanted:
        if verbose:
            string = (variable + " is not of the correct type. It is a "
                      + str(type) + " but it should be a " + str(wanted)
                      + ". ")
            if message is not None:
                string += message
            print(string)
        sys.exit()


def check_num_datasets(n_s, n_d):
    if n_s == n_d:
        if verbose:
            print("Using single dataset")
        return True
    else:
        if verbose:
            print("Using different datasets for fiducial and derivative simula\
tions")
        return False


def dataset_warning():
    if verbose:
        print("You are using your own premade dataset. Please ensure this is \
constructed correctly (see notebook). If looping then self.loop_sims=True/Fals\
e, self.input_shape and self.n_batch (and self.n_test_batch) must be set. When\
 using self.loop_sims=True indices = tf.data.Dataset.from_tensor_slices(tf.exp\
and_dims(tf.range(self.n_s, dtype=self.itype), 1)); indices = indices.batch(si\
ms_at_once).repeat(); self.indices_iterator = iter(indices) and derivative_ind\
ices = tf.data.Dataset.from_tensor_slices(tf.expand_dims(tf.range(self.n_d, dt\
ype=self.itype), 1)); derivative_indices = derivative_indices.batch(sims_at_on\
ce).repeat(); self.derivative_indices_iterator = iter(derivative_indices) must\
 be set.")


def batch_warning(n_batch):
    if n_batch is None:
        error = True
    elif type(n_batch) != int:
        error = True
    else:
        error = False
    if error:
        if verbose:
            print("Please set self.n_batch = total number of sims / number of \
sims for covariance = total number of sims for derivative / number of sims \
for derivative mean. This must be exactly divisible (int). The value provided \
is " + str(n_batch))
        sys.exit()


def loop_warning(loop_sims):
    if loop_sims is None:
        error = True
    if type(loop_sims) != bool:
        error = True
    if error:
        if verbose:
            print("Please set self.loop_sims = True/False. True must be used i\
f all the simulations do not fit in GPU memory at once. The value provided is "
                  + str(loop_sims))
        sys.exit()


def batch_warning():
    if verbose:
        print("n_batch = total number of sims / number of sims for covaria\
nce must be exactly divisible (int). Note that total number of sims for \
derivative / number of sims for derivative mean must be equal. The length of \
the input data is " + str(shape) + " and the number of sims for the covariance\
is " + str(size))
    sys.exit()


def check_numerical(type):
    if type != list:
        if verbose:
            print("Using chain rule for derivatives")
        return False
    else:
        if verbose:
            print("Using numerical derivatives")
        return True


def size_check(new, original, new_name, original_name):
    if new != original:
        if verbose:
            print("When providing " + new_name + " its size must be the be the\
 same as" + original_name + ". The size of " + new_name + " is " + str(new) +
                  " and the size of " + original_name + " is " + str(original))
        sys.exit()


def numerical_size_check(a, b, numerical):
    if numerical:
        if a != b:
            if verbose:
                print("The number of upper and lower parameter simulations for\
 the numerical derivatives should be equal. This set contains lower parameter \
simulations (first element) = " + str(a) + " and upper parameter simulations \
(second element) = " + str(b) + ". Please check this. Note that for best \
results the upper and lower parameter simulations should be seed matched and \
place correspondingly in their arrays.")
            sys.exit()
    else:
        if a != b:
            if verbose:
                print("The number of derivatives of the simulation when doing \
the derivatives analytically must be the same as the number of fiducial simula\
tions. Furthermore, these should come from corresponding seeds.")
            sys.exit()


def derivative_batch_warning(n_batch, shape, size):
    n_d_batch = shape / size
    message = ("n_batch = total number of sims / number of sims for covariance\
 must be exactly divisible (int). Note that total number of sims for \
derivative / number of sims for derivative mean must be equal. n_batch for \
the fiducial simulations = " + str(n_batch) + " and n_batch for the \
derivatives = " + str(n_d_batch) + ". The length of the data is " + str(shape)
               + " and the number of simulations for the derivatives is "
               + str(size))
    if float(int(n_d_batch)) != n_d_batch:
        if verbose:
            print(message)
        sys.exit()
    elif n_batch != int(n_d_batch):
        if verbose:
            print(message)
        sys.exit()
    else:
        return int(n_d_batch)


def length_warning(length, wanted, variable):
    if length != wanted:
        if verbose:
            print(variable + " is not of the correct length. It has length "
                  + str(length) + " but it should have length " + str(wanted))
        sys.exit()


def derivative_warning():
    if verbose:
        print("No variable has been passed for either dddθ or numerical_deriva\
tive. Either one of these must be passed. dddx should be a correctly built tf.\
data.Dataset or a numpy array with the derivative of the simulations. numerica\
l_derivative must be a either a correctly built tf.data.Dataset or a list cont\
aining two numpy arrays, [simulations at lower parameter, simulations at upper\
 parameter].")
    sys.exit()


def fiducial_check(fiducial, n_params):
    if type(fiducial) != np.ndarray:
        error = True
    elif fiducial.shape != (n_params,):
        error = True
    else:
        error = False
    if error:
        if verbose:
            print("fiducial must be a 1D numpy array of the fiducial parameter\
 values. It is a " + str(type(fiducial)) + " with values " + str(fiducial))
        sys.exit()


def delta_check(delta, n_params):
    message = "When using numerical derivatives the δθ for the derivatives mus\
t be passed as a 1D numpy array of the difference between parameter values."
    if type(delta) is not None:
        if type(delta) != np.ndarray:
            error = True
        elif delta.shape != (n_params,):
            error = True
        else:
            error = False
        if error:
            if verbose:
                print(message + " It is a " + str(type(delta))
                      + " with values " + str(delta))
            sys.exit()
    else:
        if verbose:
            print(message)
        sys.exit()


def check_model(params, summaries):
    if verbose:
        print("Checking is not currently done on the model. Make sure that its\
 output has shape " + str((None, summaries)) + " for the fiducial values and "
              + str((None, 2, params, summaries)) + " for the derivative value\
s.")
