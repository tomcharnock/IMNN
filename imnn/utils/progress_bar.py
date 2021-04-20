import tqdm.auto as tqdm
import jax
from jax.experimental.host_callback import id_tap
import jax.numpy as np


def progress_bar(max_iterations, patience, print_rate):

    if print_rate is not None:
        if max_iterations < 10000:
            pbar = tqdm.tqdm(total=max_iterations)
        else:
            pbar = tqdm.tqdm()
        remainder = max_iterations % print_rate

    def _update_pbar(args, transform):
        updates, counter, patience_counter, detF, detC, detinvC, Λ2, r = args
        pbar.update(updates)
        pbar.set_postfix(
            patience=patience_counter,
            detF=detF[counter - 1],
            detC=detC[counter - 1],
            detinvC=detinvC[counter - 1],
            Λ2=Λ2[counter - 1],
            r=r[counter - 1])

    def _close_pbar(args, transform):
        max_detF, best_w, detF, detC, detinvC, Λ2, r, \
            counter, patience_counter, state, rng = args
        updates = counter - pbar.n - 1
        pbar.update(updates)
        pbar.set_postfix(
            max_detF=max_detF,
            final_detF=detF[counter - 1],
            final_detC=detC[counter - 1],
            final_detinvC=detinvC[counter - 1],
            final_Λ2=Λ2[counter - 1],
            final_r=r[counter - 1],
            patience_reached=patience_counter)
        pbar.close()
        return False

    def _fit(func):
        def fn(x):
            return func(x)
        return fn

    def _update_controller(args):
        counter = args[0]
        jax.lax.cond(
            np.logical_and(
                np.logical_and(
                    np.greater(counter, 0),
                    np.equal(counter % print_rate, 0)),
                np.not_equal(counter, max_iterations - remainder)),
            lambda _: id_tap(
                _update_pbar, (print_rate,) + args),
            lambda _: (print_rate,) + args,
            operand=None)

        jax.lax.cond(
            np.equal(counter, max_iterations - remainder),
            lambda _: id_tap(
                _update_pbar, (remainder,) + args),
            lambda _: (remainder,) + args,
            operand=None)

    def _close_controller(cond, inputs):
        jax.lax.cond(
            cond,
            lambda _: inputs,
            lambda _: id_tap(
                _close_pbar, inputs),
            operand=None)
        return cond

    def _progress_bar(func):
        def fn(inputs):
            max_detF, best_w, detF, detC, detinvC, Λ2, r, \
                counter, patience_counter, state, rng = inputs
            _update_controller(
                (counter, patience_counter, detF, detC, detinvC, Λ2, r))
            return _close_controller(func(inputs), inputs)
        return fn

    if print_rate is not None:
        return _progress_bar
    else:
        return _fit
