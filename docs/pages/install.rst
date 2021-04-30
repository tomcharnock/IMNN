Installing
==========

The IMNN can be install by cloning the repository and installing via python or by pip installing, i.e.

.. code-block::

    git clone https://bitbucket.org/tomcharnock/imnn.git
    cd imnn
    python setup.py install

or

.. code-block::

    pip install IMNN

Notes on installation
---------------------

The IMNN was quite an early adopter of JAX and as such it uses some experimental features. It is known to be working with ``jax>=0.2.10,<=0.2.12`` and should be fine with newer versions for a while. One of the main limitations is with the use of TensorFlow Probability in the LFI module which also depends on JAX but is also dealing with the development nature of this language. The TensorFlow Datasets also requires TensorFlow>=2.1.0, but this requirement is not explicitly set so that python3.9 users can install a newer compatible version of TensorFlow without failing.

During the development of this code I implemented the value_and_jac* functions in JAX, which saves a huge amount of time for the IMNN, but these had not yet been pulled into the JAX api and as such there is a copy of these functions in ``imnn.utils.jac`` but they depend on ``jax.api`` and other functions which may change with jax development. If this becomes a problem then it will be necessary to install jax and jaxlib first, i.e. via

.. code-block::

    pip install jax==0.2.11 jaxlib==0.1.64

or whichever CUDA enabled version suits you.

The previous version of the IMNN is still available (and works well) built on a TensorFlow backend. If you want to use keras models, etc. it will probably be easier to use that. It is not as complete as this module, but is likely to be a bit more stable due to not depending on JAXs development as heavily. This can be installed via either

.. code-block::

    git clone https://bitbucket.org/tomcharnock/imnn-tf.git
    cd imnn-tf
    python setup.py install

or

.. code-block::

    pip install imnn-tf

Note that that code isn't as well documented, but there are plenty of examples still.
