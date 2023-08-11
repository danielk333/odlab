#!/usr/bin/env python

"""

"""
# import copy
import logging

import numpy as np
import pandas as pd
from astropy.time import Time
from collections import OrderedDict

logger = logging.getLogger(__name__)

POSTERIORS = OrderedDict()


def register_posterior(name):
    '''Decorator to register class as a model
    '''

    def posterior_wrapper(cls):
        logger.debug(f'Registering posterior {name}')
        assert name not in POSTERIORS, f"{name} already a registered posterior"
        POSTERIORS[name] = cls
        return cls

    return posterior_wrapper


class Posterior:

    def __init__(self, measurements, state_generator, prior=None):
        self.prior = prior

        self.state_generator = state_generator

        self.models = []
        self.dfs = []
        for model, dfs in measurements:
            self.models.append(model)
            self.dfs.append(pd.concat(dfs))

        self._reduce_and_set_df_times()

    def _reduce_and_set_df_times(self):
        """Find all the input measurements times and reduce to only unique ones 
        into a single list and set indices mapping back to the original measurements
        """
        times = np.concatenate([df["date"].values for df in self.dfs])
        utimes, indices = np.unique(times, return_inverse=True)
        self._times_df_map = np.concatenate([
            np.full((len(df),), ind, dtype=np.int64)
            for ind, df in enumerate(self.dfs)
        ])
        self._times_expand_index = indices
        self.times = Time(utimes, format="datetime64", scale="utc")

    def residuals(self, state):
        resids = []
        states = self.state_generator.get_states(state, self.times)

        for ind, (model, df) in enumerate(zip(self.models, self.dfs)):
            df_state_inds = self._times_expand_index[self._times_df_map == ind]
            sim_data = model.evaluate(df["date"], states[:, df_state_inds])

            diffs = {}
            for var in sim_data:
                diffs[var] = (df[var].values - sim_data[var])
            resids.append(diffs)
        return resids

    def logprior(self, state):
        """The log prior function"""
        if self.prior is None:
            return 0.0
        return self.prior(state)

    def logposterior(self, state):
        """The un-scaled log posterior function"""
        return self.logprior(state) + self.loglikelihood(state)

    def loglikelihood(self, state):
        raise NotImplementedError("Implement this to construct a posterior")

    def __call__(self, state):
        return self.logposterior(state)
