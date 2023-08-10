#!/usr/bin/env python

"""

"""
# import copy
import logging

from tqdm import tqdm
# import scipy.stats
import scipy.optimize as optimize
import numpy as np
import pandas as pd
from astropy.time import Time

try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
except ImportError:

    class COMM_WORLD:
        size = 1
        rank = 0

    comm = COMM_WORLD()

logger = logging.getLogger(__name__)


class MaximizeGaussianErrorPosterior:
    OPTIONS = {
        "method": "Nelder-Mead",
        "prior": None,
        "scipy-options": {},
        "bounds": None,
        "maxiter": 3000,
        "ignore_warnings": False,
    }

    def __init__(self, measurements, state_generator, **kwargs):
        self.options = {}
        self.options.update(self.OPTIONS)
        self.options.update(kwargs)

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

    # def model_jacobian(self, state0, deltas):
    #     """Calculate the observation and its numerical Jacobean
    #     of a state given the current models.
    #     """

    #     tracklets = self.data["sources"]
    #     n_sources = len(tracklets)

    #     state_, params = self._get_state_param(state0)

    #     meas_n = 0
    #     data0 = []
    #     for ind in range(n_sources):
    #         data0 += [self._models[ind].evaluate(state_, **params)]
    #         t_len = len(self._models[ind].data["t"])
    #         meas_n += t_len * len(self._models[ind].dtype)

    #     Sigma = np.zeros((meas_n,), dtype=np.float64)

    #     meas_ind0 = 0
    #     for sc_ind in range(n_sources):
    #         tn = len(self._models[sc_ind].data["t"])
    #         var_ind = 0
    #         for var, dt in self._models[sc_ind].dtype:
    #             start_ind = meas_ind0 + var_ind * tn
    #             end_ind = meas_ind0 + (var_ind + 1) * tn
    #             sig = tracklets[sc_ind].data[var + "_sd"] ** 2.0
    #             Sigma[start_ind:end_ind] = sig

    #             var_ind += 1
    #         meas_ind0 += tn * len(self._models[sc_ind].dtype)

    #     J = np.zeros([meas_n, len(self.variables)], dtype=np.float64)

    #     for ind, var in enumerate(self.variables):
    #         dstate = np.copy(state0)
    #         dstate[var][0] = dstate[var][0] + deltas[ind]

    #         dstate_, dparams = self._get_state_param(dstate)

    #         meas_ind0 = 0
    #         for sc_ind in range(n_sources):
    #             ddata = self._models[sc_ind].evaluate(dstate_, **dparams)

    #             tn = len(self._models[sc_ind].data["t"])
    #             var_ind = 0
    #             for var, dt in self._models[sc_ind].dtype:
    #                 dvar = (ddata[var] - data0[sc_ind][var]) / deltas[ind]

    #                 start_ind = meas_ind0 + var_ind * tn
    #                 end_ind = meas_ind0 + (var_ind + 1) * tn
    #                 J[start_ind:end_ind, ind] = dvar

    #                 var_ind += 1

    #             meas_ind0 += tn * len(self._models[sc_ind].dtype)

    #     return data0, J, Sigma

    # def linear_MAP_covariance(self, MAP, deltas, prior_cov_inv=None):
    #     data0, J, Sigma_m_diag = self.model_jacobian(MAP, deltas)
    #     Sigma_m_inv = np.diag(1.0 / Sigma_m_diag)

    #     if prior_cov_inv is None:
    #         Sigma_orb = np.linalg.inv(np.transpose(J) @ Sigma_m_inv @ J)
    #     else:
    #         Sigma_orb = np.linalg.inv(np.transpose(J) @ Sigma_m_inv @ J + prior_cov_inv)

    #     return Sigma_orb

    # def logprior(self, state):
    #     """The logprior function"""
    #     logprob = 0.0

    #     if self.kwargs["prior"] is None:
    #         return logprob

    #     for prior in self.kwargs["prior"]:
    #         _state = _named_to_enumerated(state, prior["variables"])
    #         dist = getattr(scipy.stats, prior["distribution"])

    #         _pr = dist.logpdf(_state, **prior["params"])
    #         if isinstance(_pr, np.ndarray):
    #             _pr = _pr[0]
    #         logprob += _pr

    #     return logprob

    # def _get_state_param(self, state):
    #     state_all = _named_to_enumerated(state, self.variables)

    #     st_var = self.kwargs["state_variables"]

    #     state_ = np.empty((len(st_var),), dtype=np.float64)
    #     params = copy.copy(self.data["params"])

    #     for sti, vari in self._variable_to_state:
    #         state_[sti] = state_all[vari]

    #     for sti, var in self._param_to_state:
    #         state_[sti] = self.data["params"][var]

    #     for par, vari in self._variable_to_param:
    #         params[par] = state_all[vari]

    #     return state_, params

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

    def loglikelihood(self, state):
        """The loglikelihood function"""
        states = self.state_generator.get_states(state, self.times)

        logsum = 0.0
        for ind, (model, df) in enumerate(zip(self.models, self.dfs)):
            df_state_inds = self._times_expand_index[self._times_df_map == ind]
            sim_data = model.evaluate(df["date"], states[:, df_state_inds])

            diffs = {}
            for var in sim_data:
                diffs[var] = (df[var].values - sim_data[var])
                logsum -= np.sum((diffs[var] / df[var + "_sd"].values) ** 2)
        return 0.5 * logsum

    def run(self, start):
        maxiter = self.options["maxiter"]

        def fun(x):
            val = self.loglikelihood(x)

            pbar.update(1)
            pbar.set_description("Posterior value = {:<10.3f} ".format(val))

            return -val

        logger.info("\n{} running {}".format(type(self).__name__, self.options["method"]))

        if self.options["ignore_warnings"]:
            np.seterr(all="ignore")

        pbar = tqdm(total=maxiter, ncols=100, position=comm.rank)
        xhat = optimize.minimize(
            fun,
            start,
            method=self.options["method"],
            options=self.options["scipy-options"],
            bounds=self.options["bounds"],
        )
        pbar.close()

        if self.options["ignore_warnings"]:
            np.seterr(all=None)

        return xhat

    # def residuals(self, state):
    #     self.loglikelihood(state)

    #     residuals = []
    #     for ind, resid in enumerate(self._tmp_residulas):
    #         residuals.append(
    #             {
    #                 "date": self._models[ind].data["date"],
    #                 "residuals": resid.copy(),
    #             }
    #         )
    #     return residuals
