#!/usr/bin/env python

"""

"""
import logging

import numpy as np

from .posterior import Posterior, register_posterior

logger = logging.getLogger(__name__)


@register_posterior("gaussian_error")
class GaussianError(Posterior):

    def model_jacobian_estimate(self, state, deltas):
        """Calculate the observation and its numerical Jacobean
        of a state given the current models.
        """
        states = self.state_generator.get_states(state, self.times)

        meas_n = np.sum([
            len(df)*len(model.OUTPUT_DATA) 
            for model, df in zip(self.models, self.dfs)
        ])
        J = np.zeros([meas_n, len(state)], dtype=np.float64)
        data0 = []
        Sigma = np.zeros((meas_n,), dtype=np.float64)
        Sigma_cursor = 0
        for ind, (model, df) in enumerate(zip(self.models, self.dfs)):
            df_state_inds = self._times_expand_index[self._times_df_map == ind]

            sim_data = model.evaluate(df["date"], states[:, df_state_inds])
            data0.append(sim_data)

            Sigma_step = len(df)
            for ind, var in enumerate(model.OUTPUT_DATA):
                Sigma[Sigma_cursor:(Sigma_cursor + Sigma_step)] = df[var + "_sd"]**2
                Sigma_cursor += Sigma_step

        for vind in range(len(state)):
            dstate = np.copy(state)
            dstate[vind] += deltas[vind]

            states = self.state_generator.get_states(dstate, self.times)

            J_cursor = 0
            for ind, (model, df) in enumerate(zip(self.models, self.dfs)):
                df_state_inds = self._times_expand_index[self._times_df_map == ind]
                sim_data = data0[ind]
                dsim_data = model.evaluate(df["date"], states[:, df_state_inds])

                J_step = len(df)
                for ind, var in enumerate(model.OUTPUT_DATA):
                    ddata = (dsim_data[var] - sim_data[var]) / deltas[ind]
                    J[J_cursor:(J_cursor + J_step), vind] = ddata
                    J_cursor += J_step

        return data0, J, Sigma

    def linear_covariance_estimate(self, state, deltas, prior_cov_inv=None):
        data0, J, Sigma_m_diag = self.model_jacobian_estimate(state, deltas)
        Sigma_m_inv = np.diag(1.0 / Sigma_m_diag)

        if prior_cov_inv is None:
            Sigma_orb = np.linalg.inv(np.transpose(J) @ Sigma_m_inv @ J)
        else:
            Sigma_orb = np.linalg.inv(np.transpose(J) @ Sigma_m_inv @ J + prior_cov_inv)

        return Sigma_orb

    def loglikelihood(self, state):
        """The log likelihood function"""
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
