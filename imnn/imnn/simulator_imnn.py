import jax
import jax.numpy as np
from functools import partial
from imnn.imnn._imnn import _IMNN
from imnn.utils import value_and_jacrev
from imnn.utils.utils import _check_simulator


class SimulatorIMNN(_IMNN):
    """Information maximising neural network fit with simulations on-the-fly

    Defines the function to get simulations and compress them using an XLA
    compilable simulator.

    The outline of the fitting procedure is that a set of :math:`i\\in[1, n_s]`
    random number generators are generated and used to generate a set of
    :math:`n_s` simulations,
    :math:`{\\bf d}^i={\\rm simulator}({\\rm seed}^i, \\theta^\\rm{fid})` at
    the fiducial model parameters, :math:`\\theta^\\rm{fid}`, and these are
    passed direrectly through a network :math:`f_{{\\bf w}}({\\bf d})` with
    network parameters :math:`{\\bf w}` to obtain network outputs
    :math:`{\\bf x}^i` and autodifferentiation is used to get the derivative of
    :math:`n_d` of these outputs with respect to the physical model parameters,
    :math:`\\partial{{\\bf x}^i}/\\partial\\theta_\\alpha`, where
    :math:`\\alpha` labels the physical parameter. With :math:`{\\bf x}^i` and
    :math:`\\partial{{\\bf x}^i}/\\partial\\theta_\\alpha` the covariance

    .. math::
        C_{ab} = \\frac{1}{n_s-1}\\sum_{i=1}^{n_s}(x^i_a-\\mu^i_a)
        (x^i_b-\\mu^i_b)

    and the derivative of the mean of the network outputs with respect to the
    model parameters

    .. math::
        \\frac{\\partial\\mu_a}{\\partial\\theta_\\alpha} = \\frac{1}{n_d}
        \\sum_{i=1}^{n_d}\\frac{\\partial{x^i_a}}{\\partial\\theta_\\alpha}

    can be calculated and used form the Fisher information matrix

    .. math::
        F_{\\alpha\\beta} = \\frac{\\partial\\mu_a}{\\partial\\theta_\\alpha}
        C^{-1}_{ab}\\frac{\\partial\\mu_b}{\\partial\\theta_\\beta}.

    The loss function is then defined as

    .. math::
        \\Lambda = -\\log|{\\bf F}| + r(\\Lambda_2) \\Lambda_2

    Since any linear rescaling of a sufficient statistic is also a sufficient
    statistic the negative logarithm of the determinant of the Fisher
    information matrix needs to be regularised to fix the scale of the network
    outputs. We choose to fix this scale by constraining the covariance of
    network outputs as

    .. math::
        \\Lambda_2 = ||{\\bf C}-{\\bf I}|| + ||{\\bf C}^{-1}-{\\bf I}||

    Choosing this constraint is that it forces the covariance to be
    approximately parameter independent which justifies choosing the covariance
    independent Gaussian Fisher information as above. To avoid having a dual
    optimisation objective, we use a smooth and dynamic regularisation strength
    which turns off the regularisation to focus on maximising the Fisher
    information when the covariance has set the scale

    .. math::
        r(\\Lambda_2) = \\frac{\\lambda\\Lambda_2}{\\Lambda_2-\\exp
        (-\\alpha\\Lambda_2)}.

    Once the loss function is calculated the automatic gradient is then
    calculated and used to update the network parameters via the optimiser
    function.

    Methods
    -------
    simulator:
        A function for generating a simulation on-the-fly (XLA compilable)

    """
    def __init__(self, n_s, n_d, n_params, n_summaries, input_shape, θ_fid,
                 model, optimiser, key_or_state, simulator):
        """Constructor method

        Initialises all IMNN attributes, constructs neural network and its
        initial parameter values and creates history dictionary. Also checks
        validity of simulator and sets the ``simulate`` attribute to ``True``.

        Parameters
        ----------
        n_s : int
            Number of simulations used to calculate summary covariance
        n_d : int
            Number of simulations used to calculate mean of summary derivative
        n_params : int
            Number of model parameters
        n_summaries : int
            Number of summaries, i.e. outputs of the network
        input_shape : tuple
            The shape of a single input to the network
        θ_fid : float(n_params,)
            The value of the fiducial parameter values used to generate inputs
        model : tuple, len=2
            Tuple containing functions to initialise neural network
            ``fn(rng: int(2), input_shape: tuple) -> tuple, list`` and the
            neural network as a function of network parameters and inputs
            ``fn(w: list, d: float(None, input_shape)) -> float(None, n_summari
            es)``.
            (Essentibly stax-like, see `jax.experimental.stax <https://jax.read
            thedocs.io/en/stable/jax.experimental.stax.html>`_))
        optimiser : tuple, len=3
            Tuple containing functions to generate the optimiser state
            ``fn(x0: list) -> :obj:state``, to update the state from a list of
            gradients ``fn(i: int, g: list, state: :obj:state) -> :obj:state``
            and to extract network parameters from the state
            ``fn(state: :obj:state) -> list``.
            (See `jax.experimental.optimizers <https://jax.readthedocs.io/en/st
            able/jax.experimental.optimizers.html>`_)
        key_or_state : int(2) or :obj:state
            Either a stateless random number generator or the state object of
            an preinitialised optimiser
        simulator : fn
            A function that generates a single simulation from a random number
            generator and a tuple (or array) of parameter values at which to
            generate the simulations. For the purposes of use in LFI/ABC
            afterwards it is also useful for the simulator to be able to
            broadcast to a batch of simulations on the zeroth axis
            ``fn(int(2,), float([None], n_params)) ->
            float([None], input_shape)``
        """
        super().__init__(
            n_s=n_s,
            n_d=n_d,
            n_params=n_params,
            n_summaries=n_summaries,
            input_shape=input_shape,
            θ_fid=θ_fid,
            model=model,
            key_or_state=key_or_state,
            optimiser=optimiser)
        self.simulator = _check_simulator(simulator)
        self.simulate = True

    def _get_fitting_keys(self, rng):
        """Generates random numbers for simulation

        Parameters
        ----------
        rng : int(2,)
            A random number generator

        Returns
        -------
        int(2,), int(2,), int(2,)
            A new random number generator and random number generators for
            fitting (and validation)
        """
        return jax.random.split(rng, num=3)

    def get_summaries(self, w, key, validate=False):
        """Gets all network outputs and derivatives wrt model parameters

        A random seed for each simulation is obtained and ``n_d`` of them are
        used to calculate the network outputs of each of these simulations as
        well as the derivative of these network outputs with respect to the
        model parameters as calculated using jax autodifferentiation. The
        remaining ``n_s - n_d`` network outputs are then calculated and
        concatenated to those already calculated.

        Parameters
        ----------
        w : list or None, default=None
            The network parameters if wanting to calculate the Fisher
            information with a specific set of network parameters
        key : int(2,) or None, default=None
            A random number generator for generating simulations on-the-fly
        validate : bool, default=False
            Whether to get summaries of the validation set

        Returns
        -------
        float(n_s, n_summaries):
            The set of all network outputs used to calculate the covariance
        float(n_d, n_summaries, n_params):
            The set of all network output derivatives wrt model parameters

        Methods
        -------
        get_summary:
            Return a single network output
        get_derivatives:
            Return the Jacobian of the network outputs wrt model parameters
        """
        def get_summary(key, θ):
            """Return a single network output

            Parameters
            ----------
            key : int(2,)
                A random number generator for generating simulations
            θ : float(n_params,)
                The value of the model parameters to generate the simulation at

            Returns
            -------
            float(n_summaries):
                A single simulation passed through the neural network
            """
            return self.model(w, self.simulator(key, θ))

        def get_derivatives(key):
            """Return the Jacobian of the network outputs wrt model parameters

            Parameters
            ----------
            key : int(2,)
                A random number generator for generating simulations

            Returns
            -------
            float(n_summaries):
                A single simulation passed through the neural network
            float(n_summaries, n_params)
                The derivative of the network output wrt model parameters
            """
            return value_and_jacrev(get_summary, argnums=1)(key, self.θ_fid)

        keys = np.array(jax.random.split(key, num=self.n_s))
        summaries, derivatives = jax.vmap(get_derivatives)(keys[:self.n_d])
        if self.n_s > self.n_d:
            summaries = np.vstack([
                summaries,
                jax.vmap(partial(get_summary, θ=self.θ_fid))(keys[self.n_d:])])
        return summaries, derivatives
