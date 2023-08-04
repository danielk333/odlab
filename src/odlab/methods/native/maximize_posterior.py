#!/usr/bin/env python

"""

"""
# import copy

# from tqdm import tqdm
# import scipy.stats
# import scipy.optimize as optimize
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


class MaximizeGaussianErrorPosterior:
    OPTIONS = {
        "method": "Nelder-Mead",
        "prior": None,
        "scipy-options": {},
        "bounds": None,
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

        times = pd.concat([df["date"] for df in dfs]).values
        self.indexing = np.stack([df.index for df in dfs])
        self.times = Time(times, format="datetime64", scale="utc")
        breakpoint()

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

    def loglikelihood(self, state):
        """The loglikelihood function"""
        states = self.state_generator.get_states(state, self.times)
        
        for model, df in zip(self.models, self.dfs):
            
            sim_data = model.evaluate(self.times, states)

        # logsum = 0.0
        # for ind in range(n_tracklets):
        #     sim_data = self._models[ind].evaluate(state_, **params)
        #     _residuals = self._models[ind].distance(
        #         sim_data,
        #         tracklets[ind].data,
        #     )
        #     for name, _ in self._models[ind].dtype:
        #         self._tmp_residulas[ind][name] = _residuals[name]

        #     num = len(self._models[ind].data["t"])

        #     for name, _ in self._models[ind].dtype:
        #         tr_var = tracklets[ind].data[name + "_sd"] ** 2.0
        #         logsum += np.sum(-1.0 * _residuals[name] ** 2.0 / tr_var)

        # return 0.5 * logsum

    # def run(self):
    #     if self.kwargs["start"] is None and self.kwargs["prior"] is None:
    #         raise ValueError("No start value or prior given.")

    #     start = _named_to_enumerated(self.kwargs["start"], self.variables)

    #     maxiter = self.kwargs["options"].get("maxiter", 3000)

    #     def fun(x):
    #         _x = _enumerated_to_named(x, self.variables)

    #         try:
    #             val = self.evalute(_x)
    #         except Exception:
    #             val = -np.inf
    #             raise

    #         pbar.update(1)
    #         pbar.set_description("Least Squares = {:<10.3f} ".format(-val))

    #         return -val

    #     print("\n{} running {}".format(type(self).__name__, self.kwargs["method"]))

    #     pbars = []
    #     for pbar_id in range(comm.size):
    #         pbars.append(tqdm(total=maxiter, ncols=100))
    #     pbar = pbars[comm.rank]
    #     for ind in range(comm.size):
    #         if ind != comm.rank:
    #             pbars[ind].close()

    #     xhat = optimize.minimize(
    #         fun,
    #         start,
    #         method=self.kwargs["method"],
    #         options=self.kwargs["scipy-options"],
    #         bounds=self.kwargs["bounds"],
    #     )
    #     pbar.close()

    #     self.results.trace = _enumerated_to_named(xhat.x, self.variables)
    #     self.results.MAP = self.results.trace.copy()
    #     self.loglikelihood(self.results.MAP)
    #     self.results.residuals = copy.deepcopy(self._tmp_residulas)
    #     self.results.date = self.data["date0"]

    #     return self.results

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
