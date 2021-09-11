from typing import Callable, Union, Sequence
from jax._src.api import _check_callable, _check_arg, vmap, \
    _jvp, _check_input_dtype_jacfwd, _check_output_dtype_jacfwd, \
    _vjp, _check_input_dtype_jacrev, _check_output_dtype_jacrev, \
    _std_basis, _unravel_array_into_pytree
from jax.api_util import argnums_partial, _ensure_index
from jax.tree_util import tree_map, tree_structure, tree_transpose
from jax._src.util import partial, wraps
from jax._src.traceback_util import api_boundary
import numpy as np
import jax.linear_util as lu
import jax.core as core
import jax.dtypes as dtypes

def jacfwd(fun: Callable, argnums: Union[int, Sequence[int]] = 0,
           holomorphic: bool = False) -> Callable:
  """Creates a function returning Jacobian of ``fun`` evaluated column-by-column
     using forward-mode AD.

  Args:
    fun: Function whose Jacobian is to be computed.
    argnums: Optional, integer or sequence of integers. Specifies which
      positional argument(s) to differentiate with respect to (default ``0``).
    holomorphic: Optional, bool. Indicates whether ``fun`` is promised to be
      holomorphic. Default False.

  Returns:
    A function with the same arguments as ``fun``, that evaluates the Jacobian of
    ``fun`` using forward-mode automatic differentiation.

  For example:

  >>> import jax
  >>> import jax.numpy as jnp
  >>>
  >>> def f(x):
  ...   return jnp.asarray(
  ...     [x[0], 5*x[2], 4*x[1]**2 - 2*x[2], x[2] * jnp.sin(x[0])])
  ...
  >>> print(jax.jacfwd(f)(jnp.array([1., 2., 3.])))
  [[ 1.       0.       0.     ]
   [ 0.       0.       5.     ]
   [ 0.      16.      -2.     ]
   [ 1.6209   0.       0.84147]]
  """
  value_and_jacfwd_f = value_and_jacfwd(fun, argnums, holomorphic=holomorphic)

  docstr = ("Foward-mode Jacobian of {fun} with respect to positional "
            "argument(s) {argnums}.")

  @wraps(fun, docstr=docstr, argnums=argnums)
  @api_boundary
  def jacfwd_f(*args, **kwargs):
    _, jf = value_and_jacfwd_f(*args, **kwargs)
    return jf

  return jacfwd_f

def value_and_jacfwd(fun: Callable, argnums: Union[int, Sequence[int]] = 0,
                     holomorphic: bool = False) -> Callable:
  """Jacobian of ``fun`` evaluated column-by-column using forward-mode AD.

  Args:
    fun: Function whose Jacobian is to be computed.
    argnums: Optional, integer or sequence of integers. Specifies which
      positional argument(s) to differentiate with respect to (default ``0``).
    holomorphic: Optional, bool. Indicates whether ``fun`` is promised to be
      holomorphic. Default False.

  Returns:
    The value of the function to be automatically differentiated.
    A function with the same arguments as ``fun``, that evaluates the Jacobian of
    ``fun`` using forward-mode automatic differentiation.

  For example:

  >>> import jax
  >>> import jax.numpy as jnp
  >>>
  >>> def f(x):
  ...   return jnp.asarray(
  ...     [x[0], 5*x[2], 4*x[1]**2 - 2*x[2], x[2] * jnp.sin(x[0])])
  ...
  >>> print(jax.value_and_jacfwd(f)(jnp.array([1., 2., 3.]))[0])
  [ 1.        15.        10.         2.5244129]
  >>> print(jax.value_and_jacfwd(f)(jnp.array([1., 2., 3.]))[1])
  [[ 1.          0.          0.        ]
   [ 0.          0.          5.        ]
   [ 0.         16.         -2.        ]
   [ 1.6209068   0.          0.84147096]]
  """
  docstr = ("Value and forward-mode Jacobian of {fun} with respect to "
            "positional argument(s) {argnums}. Takes the same arguments as "
            "{fun} but returns a two-element tuple where the first element "
            "is the value of {fun} and the second element is the Jacobian.")

  _check_callable(fun)
  argnums = _ensure_index(argnums)

  @wraps(fun, docstr=docstr, argnums=argnums)
  @api_boundary
  def value_and_jacfwd_f(*args, **kwargs):
    f = lu.wrap_init(fun, kwargs)
    f_partial, dyn_args = argnums_partial(f, argnums, args)
    tree_map(partial(_check_input_dtype_jacfwd, holomorphic), dyn_args)
    pushfwd = partial(_jvp, f_partial, dyn_args)
    y, jac = vmap(pushfwd, out_axes=(None, -1))(_std_basis(dyn_args))
    tree_map(partial(_check_output_dtype_jacfwd, holomorphic), y)
    example_args = dyn_args[0] if isinstance(argnums, int) else dyn_args
    return y, tree_map(partial(_unravel_array_into_pytree, example_args, -1), jac)

  return value_and_jacfwd_f

def _check_input_dtype_jacfwd(holomorphic, x):
  _check_arg(x)
  aval = core.get_aval(x)
  if holomorphic:
    if not (dtypes.issubdtype(aval.dtype, np.complexfloating) and
            not dtypes.issubdtype(aval.dtype, np.floating)):
      raise TypeError("jacfwd with holomorphic=True requires inputs with complex dtype, "
                      f"but got {aval.dtype.name}.")
  elif not dtypes.issubdtype(aval.dtype, np.floating):
    raise TypeError("jacfwd requires real-valued inputs (input dtype that is "
                    f"a sub-dtype of np.floating), but got {aval.dtype.name}. "
                    "For holomorphic differentiation, pass holomorphic=True. "
                    "For differentiation of non-holomorphic functions involving complex "
                    "inputs or integer inputs, use jax.jvp directly.")

def _check_output_dtype_jacfwd(holomorphic, x):
  aval = core.get_aval(x)
  if holomorphic:
    if not dtypes.issubdtype(aval.dtype, np.complexfloating):
      raise TypeError("jacfwd with holomorphic=True requires outputs with complex dtype, "
                      f"but got {aval.dtype.name}.")


def jacrev(fun: Callable, argnums: Union[int, Sequence[int]] = 0,
           has_aux: bool = False, holomorphic: bool = False,
           allow_int: bool = False) -> Callable:
  """Creates a function which evaluates the Jacobian of ``fun`` evaluated
     row-by-row using reverse-mode AD.

  Args:
    fun: Function whose Jacobian is to be computed.
    argnums: Optional, integer or sequence of integers. Specifies which
      positional argument(s) to differentiate with respect to (default ``0``).
    has_aux: Optional, bool. Indicates whether ``fun`` returns a pair where the
      first element is considered the output of the mathematical function to be
      differentiated and the second element is auxiliary data. Default False.
    holomorphic: Optional, bool. Indicates whether ``fun`` is promised to be
      holomorphic. Default False.
    allow_int: Optional, bool. Whether to allow differentiating with
      respect to integer valued inputs. The gradient of an integer input will
      have a trivial vector-space dtype (float0). Default False.

  Returns:
    A function with the same arguments as ``fun``, that evaluates the Jacobian of
    ``fun`` using reverse-mode automatic differentiation. If ``has_aux`` is True
    then a pair of (Jacobian, auxiliary_data) is returned.

  For example:

  >>> import jax
  >>> import jax.numpy as jnp
  >>>
  >>> def f(x):
  ...   return jnp.asarray(
  ...     [x[0], 5*x[2], 4*x[1]**2 - 2*x[2], x[2] * jnp.sin(x[0])])
  ...
  >>> print(jax.jacrev(f)(jnp.array([1., 2., 3.])))
  [[ 1.       0.       0.     ]
   [ 0.       0.       5.     ]
   [ 0.      16.      -2.     ]
   [ 1.6209   0.       0.84147]]
  """
  value_and_jacrev_f = value_and_jacrev(fun, argnums, has_aux=has_aux,
                                        holomorphic=holomorphic,
                                        allow_int=allow_int)

  docstr = ("Reverse-mode Jacobian of {fun} with respect to positional "
          "argument(s) {argnums}.")

  @wraps(fun, docstr=docstr, argnums=argnums)
  @api_boundary
  def jacrev_f(*args, **kwargs):
    _, jr = value_and_jacrev_f(*args, **kwargs)
    return jr

  @wraps(fun, docstr=docstr, argnums=argnums)
  @api_boundary
  def jacrev_f_aux(*args, **kwargs):
    (_, aux), jr = value_and_jacrev_f(*args, **kwargs)
    return jr, aux

  return jacrev_f_aux if has_aux else jacrev_f

def value_and_jacrev(fun: Callable, argnums: Union[int, Sequence[int]] = 0,
                     has_aux: bool = False, holomorphic: bool = False,
                     allow_int: bool = False) -> Callable:
  """Jacobian of ``fun`` evaluated row-by-row using reverse-mode AD.

  Args:
    fun: Function whose Jacobian is to be computed.
    argnums: Optional, integer or sequence of integers. Specifies which
      positional argument(s) to differentiate with respect to (default ``0``).
    has_aux: Optional, bool. Indicates whether ``fun`` returns a pair where the
      first element is considered the output of the mathematical function to be
      differentiated and the second element is auxiliary data. Default False.
    holomorphic: Optional, bool. Indicates whether ``fun`` is promised to be
      holomorphic. Default False.
    allow_int: Optional, bool. Whether to allow differentiating with
      respect to integer valued inputs. The gradient of an integer input will
      have a trivial vector-space dtype (float0). Default False.

  Returns:
    The value of the function to be automatically differentiated.
    A function with the same arguments as ``fun``, that evaluates the Jacobian of
    ``fun`` using reverse-mode automatic differentiation.

  For example:

  >>> import jax
  >>> import jax.numpy as jnp
  >>>
  >>> def f(x):
  ...   return jnp.asarray(
  ...     [x[0], 5*x[2], 4*x[1]**2 - 2*x[2], x[2] * jnp.sin(x[0])])
  ...
  >>> print(jax.value_and_jacrev(f)(jnp.array([1., 2., 3.]))[0])
  [ 1.        15.        10.         2.5244129]
  >>> print(jax.value_and_jacrev(f)(jnp.array([1., 2., 3.]))[1])
  [[ 1.          0.          0.        ]
   [ 0.          0.          5.        ]
   [ 0.         16.         -2.        ]
   [ 1.6209068   0.          0.84147096]]
  """

  docstr = ("Value and reverse-mode Jacobian of {fun} with respect to "
            "positional argument(s) {argnums}. Takes the same arguments as "
            "{fun} but returns a two-element tuple where the first element "
            "is the value of {fun} and the second element is the Jacobian.")

  _check_callable(fun)

  @wraps(fun, docstr=docstr, argnums=argnums)
  @api_boundary
  def value_and_jacrev_f(*args, **kwargs):
    f = lu.wrap_init(fun, kwargs)
    f_partial, dyn_args = argnums_partial(f, argnums, args)
    tree_map(partial(_check_input_dtype_jacrev, holomorphic, allow_int), dyn_args)
    if not has_aux:
      y, pullback = _vjp(f_partial, *dyn_args)
    else:
      y, pullback, aux = _vjp(f_partial, *dyn_args, has_aux=True)
    tree_map(partial(_check_output_dtype_jacrev, holomorphic), y)
    jac = vmap(pullback)(_std_basis(y))
    jac = jac[0] if isinstance(argnums, int) else jac
    example_args = dyn_args[0] if isinstance(argnums, int) else dyn_args
    jac = tree_map(partial(_unravel_array_into_pytree, y, 0), jac)
    if not has_aux:
      return y, tree_transpose(tree_structure(example_args), tree_structure(y), jac)
    else:
      return (y, aux), tree_transpose(tree_structure(example_args), tree_structure(y), jac)
    return

  return value_and_jacrev_f
