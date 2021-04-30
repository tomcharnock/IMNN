Welcome to IMNN's documentation!
================================

The information maximising neural network (IMNN) is a fitting algorithm for neural networks that aims to maximise the Fisher information of a training set to produce the most informative set of summaries about the model parameters of a generative model of some target data. In particular, an IMNN can extract, asymptotically losslessly, information from complex distributed data in a :math:`d`-dimensional space and map it to some normally distributed summaries in a :math:`n_\rm{params}`-dimensional space, where :math:`n_\rm{params}` is the number of parameters in the generative model for the data.

|doc| |pypi| |bit| |git| |doi| |zen|

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   pages/details
   pages/install
   pages/examples
   pages/modules
   pages/parents
   pages/tf.data.Datasets
   pages/lfi


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. |doc| image:: /_images/doc.svg
    :target: https://www.aquila-consortium.org/doc/imnn/

.. |pypi| image:: /_images/pypi.svg
    :target: https://pypi.org/project/IMNN/

.. |bit| image:: /_images/bit.svg
    :target: https://bitbucket.org/tomcharnock/imnn

.. |git| image:: /_images/git.svg
    :target: https://github.com/tomcharnock/imnn

.. |doi| image:: https://zenodo.org/badge/DOI/10.1103/PhysRevD.97.083004.svg
   :target: https://doi.org/10.1103/PhysRevD.97.083004

.. |zen| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1175196.svg
   :target: https://doi.org/10.5281/zenodo.1175196
