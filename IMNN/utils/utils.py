"""A bunch of utility functions for checking stuff in the IMNN

TODO
____
Docstrings need writing
"""


__version__ = '0.2a4'
__author__ = "Tom Charnock"


import sys
import numpy as np


class utils():
    def __init__(self, verbose=True):
        self.verbose = True
        self.verbose = self.type_checking(verbose, True, "verbose")

    def type_checking(self, value, wanted, variable, message=None):
        if value is None:
            if self.verbose:
                print(variable + " cannot be None")
            sys.exit()
        if type(value) != type(wanted):
            if self.verbose:
                string = (variable + " is not of the correct type. It is a "
                          + str(type(value)) + " but it should be a "
                          + str(type(wanted)) + ". ")
                if message is not None:
                    string += message
                print(string)
            sys.exit()
        return value

    def positive_integer(self, value, variable):
        value = self.type_checking(value, 1, variable,
                                   message="It should also be positive.")
        if value < 1:
            if self.verbose:
                print((variable
                       + " must be a positive integer but has a value of "
                       + str(value)))
            sys.exit()
        return value

    def check_shape(self, value, type, shape, variable):
        value = self.type_checking(
            value, type, variable,
            message="It should also have shape " + str(shape) + ".")
        if value.shape != shape:
            if self.verbose:
                print((variable
                       + " should have shape "
                       + str(shape)
                       + " but has a value of "
                       + str(value.shape)))
            sys.exit()
        return value

    def data_error(self, validate=False):
        if self.verbose:
            if validate:
                print("Both validatation_fiducial_loader and \
validation_derivative_loader must be either numpy arrays OR callable functions\
 to load the data into the dataset and be of the same type as the training \
 data.")
            else:
                print("Both fiducial_loader and derivative_loader must be \
either numpy arrays OR callable functions to load the data into the dataset")
        sys.exit()

    def regularisation_error(self):
        if self.verbose:
            if validate:
                print("λ and ϵ must be passed to set the regularisation rate \
and strength.")
        sys.exit()

    def save_error(self):
        if self.verbose:
            print("Need to save model for patience to work.\n" +
                  "Run IMNN.save=True;\n" +
                  "IMNN.filename='save-directory-path';\n" +
                  "IMNN.model.save(IMNN.filename)")
        sys.exit()

    def check_model(self, model, input_shape, output_shape):
        if not hasattr(model, "input_shape"):
            if self.verbose:
                print("model must have an input_shape attribute")
            sys.exit()
        if not hasattr(model, "output_shape"):
            if self.verbose:
                print("model must have an output_shape attribute")
            sys.exit()
        if not hasattr(model, "save"):
            if self.verbose:
                print("model must have an save function attribute")
            sys.exit()
        if not hasattr(model, "save_weights"):
            if self.verbose:
                print("model must have an save_weights function attribute")
            sys.exit()
        if not hasattr(model, "reset_states"):
            if self.verbose:
                print("model must have an reset_states function attribute")
            sys.exit()
        if model.input_shape[1:] != input_shape:
            if self.verbose:
                print("the model has an input shape of "
                      + str(model.input_shape[1:])
                      + " but the data has shape "
                      + str(input_shape)
                      + ". Cannot continue.")
            sys.exit()
        if model.output_shape[1:] != (output_shape,):
            if self.verbose:
                print("the model has an output shape of "
                      + str(model.output_shape[1:])
                      + " but the summary shape must be "
                      + str((output_shape,))
                      + ". Cannot continue.")
            sys.exit()
        return model

    def at_once_checker(self, value, n_s, n_d, n_params):
        value = self.positive_integer(value, "at_once")
        if value > n_s:
            if self.verbose:
                print("at_once is greater than n_s - setting to n_s. You \
should consider uploading data as a single tensor.")
            fiducial_at_once = n_s
        else:
            fiducial_at_once = value
        if value > n_d * n_params * 2:
            if self.verbose:
                print("at_once is greater than n_d * n_params * 2 - setting to\
 n_d * n_params * 2.")
            derivative_at_once = n_d * n_params * 2
        elif value > n_d:
            if self.verbose:
                print("at_once is greater than n_d - setting at_once to n_d \
for derivatives")
            derivative_at_once = n_d
        else:
            derivatives_at_once = value
        return fiducial_at_once, derivative_at_once


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
