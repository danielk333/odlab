#!/usr/bin/env python

"""

"""
from tqdm import tqdm
import numpy as np

try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
except ImportError:

    class COMM_WORLD:
        size = 1
        rank = 0

    comm = COMM_WORLD()

from .solvers import Solver, register_solver


@register_solver("mcmc_scam")
class Scam(Solver):
    """Markov Chain Monte Carlo sampling of the posterior,
    assuming all measurement errors are Gaussian (thus the log likelihood
    becomes a least squares).
    """

    OPTIONS = {
        "accept_max": 0.5,
        "accept_min": 0.3,
        "adapt_interval": 1000,
        "proposal_adapt_interval": 10000,
        "tune": 1000,
        "proposal": "normal",
        "jacobian_delta": 0.1,
        "progress_bar": True,
    }

    def __init__(self, base_step_size, **kwargs):
        self.base_step_size = base_step_size
        super().__init__(**kwargs)

    def run(self, posterior, start, steps, seed=None):
        if seed is not None:
            seed_state = np.random.seed()
            np.random.seed(seed)
        xnow = np.copy(start)
        step = np.copy(self.base_step_size)

        n_var = len(start)

        run_steps = self.options["tune"] + steps
        chain = np.empty((n_var, run_steps), dtype=np.float64)

        logpost = posterior(xnow)

        accept = np.zeros((n_var,), dtype=np.float64)
        tries = np.zeros((n_var,), dtype=np.float64)

        if self.options["proposal"] in ["normal", "adaptive"]:
            proposal_mu = np.zeros((n_var,), dtype=np.float64)
            proposal_cov = np.eye(n_var, dtype=np.float64)
            proposal_axis = np.eye(n_var, dtype=np.float64)

        elif self.options["proposal"] == "LinSigma":
            deltas = self.options["jacobian_delta"]
            if not isinstance(deltas, np.ndarray):
                deltas = np.ones((n_var,), dtype=np.float64) * deltas

            Sigma_orb = posterior.linear_covariance_estimate(start, deltas, prior_cov_inv=None)

            proposal_mu = np.zeros((n_var,), dtype=np.float64)
            eigs, proposal_axis = np.linalg.eig(Sigma_orb)
            proposal_cov = np.diag(eigs)
        else:
            raise ValueError(
                f'proposal option "{self.options["proposal"]}"\
                not recognized'
            )

        if self.options["progress_bar"]:
            pbar = tqdm(position=comm.rank, total=run_steps)

        for ind in range(run_steps):
            if self.options["progress_bar"]:
                pbar.update(1)
                pbar.set_description("Sampling log-posterior = {:<10.3f} ".format(logpost))

            xtry = np.copy(xnow)

            pi = int(np.floor(np.random.rand(1) * n_var))

            proposal0 = np.random.multivariate_normal(proposal_mu, proposal_cov)
            proposal = proposal0[pi] * proposal_axis[:, pi]

            xtry += proposal * step[pi]

            logpost_try = posterior(xtry)
            alpha = np.log(np.random.rand(1))

            if logpost_try > logpost:
                _accept = True
            elif (logpost_try - alpha) > logpost:
                _accept = True
            else:
                _accept = False

            tries[pi] += 1.0

            if _accept:
                logpost = logpost_try
                xnow = xtry
                accept[pi] += 1.0

            ad_inv = self.options["adapt_interval"]
            cov_ad_inv = self.options["proposal_adapt_interval"]
            ac_min = self.options["accept_min"]
            ac_max = self.options["accept_max"]
            if ind % ad_inv == 0 and ind > 0:
                for var_ind in range(n_var):
                    ratio = accept[var_ind] / tries[var_ind]

                    if ratio > ac_max:
                        step[var_ind] *= 2.0
                    elif ratio < ac_min:
                        step[var_ind] /= 2.0

                    accept[var_ind] = 0.0
                    tries[var_ind] = 0.0

            if (ind % cov_ad_inv == 0 and ind > 0 and self.kwargs["proposal"] == "adaptive"):
                _data = chain[var_ind, :ind]
                _proposal_cov = np.corrcoef(_data)

                if not np.any(np.isnan(_proposal_cov)):
                    eigs, proposal_axis = np.linalg.eig(_proposal_cov)
                    proposal_cov = np.diag(eigs)

            chain[:, ind] = xnow

        if seed is not None:
            np.random.seed(seed_state)

        return chain[self.options["tune"]:]
