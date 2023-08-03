#!/usr/bin/env python

'''

'''
import logging
import numpy as np
from tqdm import tqdm

from . import times

logger = logging.getLogger(__name__)


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
            times.npdt2mjd(date0),
            **params
        )
        states[it] = unstructured_to_structured(
            prop_state.T, results.trace.dtype)
        it += 1
        pbar.update(1)

    return states, num
