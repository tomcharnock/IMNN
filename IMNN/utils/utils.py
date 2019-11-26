"""A bunch of utility functions for checking stuff in the IMNN

TODO
____
Docstrings need writing
"""


__version__ = '0.2a1'
__author__ = "Tom Charnock"


import sys
import numpy as np


class utils():
    def __init__(self, verbose=True):
        self.verbose = verbose

    def positive_integer(self, param, name):
        if type(param) != int:
            error = True
        elif param < 1:
            error = True
        else:
            error = False
            if error:
                if self.verbose:
                    print(name
                          + " must be a positive integer but has a value of "
                          + str(param))
                sys.exit()
            else:
                return param

    def bool_none(self, param):
        if param is not None:
            a = True
        else:
            a = False
        return a

    def type_warning(self, type, wanted, variable, message=None):
        if type is None:
            if self.verbose:
                print(variable + " cannot be None")
            sys.exit()
        if type != wanted:
            if self.verbose:
                string = (variable + " is not of the correct type. It is a "
                          + str(type) + " but it should be a " + str(wanted)
                          + ". ")
                if message is not None:
                    string += message
                print(string)
            sys.exit()

    def check_num_datasets(self, n_s, n_d):
        if n_s == n_d:
            if self.verbose:
                print("Using single dataset")
            return True
        else:
            if self.verbose:
                print("Using different datasets for fiducial and derivative\
simulations")
            return False

    def batch_warning(self, n_batch):
        if n_batch is None:
            error = True
        elif type(n_batch) != int:
            error = True
        else:
            error = False
        if error:
            if self.verbose:
                print("Please set self.n_batch = total number of sims / number\
 of sims for covariance = total number of sims for derivative / number of sims\
 for derivative mean. This must be exactly divisible (int). The value provided\
 is " + str(n_batch))
            sys.exit()

    def batch_warning(self, shape, size, train, derivative=False):
        n_batch = shape / size
        if float(int(n_batch)) == n_batch:
            return int(n_batch)
        else:
            if self.verbose:
                print("n_batch = total number of sims / number of sims for cov\
ariance must be exactly divisible (int). Note that total number of sims for \
derivative / number of sims for derivative mean must be equal. The length of \
the input data is " + str(shape) + " and the number of sims for the covariance\
is " + str(size))
            sys.exit()

    def size_check(self, new, original, new_name, original_name):
        if new != original:
            if self.verbose:
                print("When providing " + new_name + " its size must be the be\
 the same as" + original_name + ". The size of " + new_name + " is " + str(new)
                      + " and the size of " + original_name + " is "
                      + str(original))
            sys.exit()

    def numerical_size_check(self, a, b, numerical):
        if numerical:
            if a != b:
                if self.verbose:
                    print("The number of upper and lower parameter simulations\
 for the numerical derivatives should be equal. This set contains lower parame\
ter simulations (first element) = " + str(a) + " and upper parameter simulatio\
ns (second element) = " + str(b) + ". Please check this. Note that for best \
results the upper and lower parameter simulations should be seed matched and \
place correspondingly in their arrays.")
                sys.exit()
        else:
            if a != b:
                if self.verbose:
                    print("The number of derivatives of the simulation when do\
ing the derivatives analytically must be the same as the number of fiducial si\
mulations. Furthermore, these should come from corresponding seeds.")
                sys.exit()

    def fiducial_check(self, fiducial, n_params):
        if type(fiducial) != np.ndarray:
            error = True
        elif fiducial.shape != (n_params,):
            error = True
        else:
            error = False
        if error:
            if self.verbose:
                print("fiducial must be a 1D numpy array of the fiducial param\
eter values. It is a " + str(type(fiducial)) + " with values " + str(fiducial))
            sys.exit()

    def delta_check(self, delta, n_params):
        message = "When using numerical derivatives the δθ for the derivatives\
 must be passed as a 1D numpy array of the difference between parameter values"
        if type(delta) is not None:
            if type(delta) != np.ndarray:
                error = True
            elif delta.shape != (n_params,):
                error = True
            else:
                error = False
            if error:
                if self.verbose:
                    print(message + " It is a " + str(type(delta))
                          + " with values " + str(delta))
                sys.exit()
        else:
            if self.verbose:
                print(message)
            sys.exit()

    def check_model(self, params, summaries):
        if self.verbose:
            print("Checking is not currently done on the model. Make sure that\
 its output has shape " + str((None, summaries)) + " for the fiducial values a\
 nd "
                  + str((None, 2, params, summaries)) + " for the derivative v\
alues.")

    def isnotebook(self):
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
