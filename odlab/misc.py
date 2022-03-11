#!/usr/bin/env python

'''

'''

# Python standard import

# Third party import
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
from numpy.lib.recfunctions import unstructured_to_structured
from tqdm import tqdm

# Local import
from . import datetime as datetime_local


def propagate_results(t, date0, results, propagator, num=None, params=None):
    if params is None:
        params = {}
    if num is None:
        num = range(len(results.trace))
    else:
        num = np.random.randint(len(results.trace), size=num)

    pbar = tqdm(total=len(num), ncols=100)

    states = np.empty((len(num),), dtype=results.trace.dtype)

    it = 0
    for i in num:
        state = structured_to_unstructured(
            results.trace[i][['x', 'y', 'z', 'vx', 'vy', 'vz']])

        prop_state = propagator.propagate(
            np.array([t]),
            state,
            datetime_local.npdt2mjd(date0),
            **params
        )
        states[it] = unstructured_to_structured(
            prop_state.T, results.trace.dtype)
        it += 1
        pbar.update(1)

    return states, num
