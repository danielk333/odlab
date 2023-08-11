#!/usr/bin/env python

"""

Run the "simulate_radar_data.py" before trying this

"""
import pathlib

import numpy as np

# import matplotlib.pyplot as plt
from astropy.time import Time

# import pandas as pd

import odlab
import odlab.methods as methods
import sorts
import pyorb

print("Solvers:\n", odlab.SOLVERS)
print("Posteriors:\n", odlab.POSTERIORS)

np.random.seed(897824)

data_dir = (pathlib.Path(__file__).parent / "tmp").resolve()

dfs = odlab.glob_sources(data_dir, {"radar_hdf": "sim_radar*.h5"})
assert len(dfs) > 0, "Run the 'simulate_radar_data.py' before trying this"

state0_true = np.load(data_dir / "state0.npy")
state0 = state0_true.copy()
state0[:3, :] += np.random.randn(3, 1) * 1e3
state0[3:, :] += np.random.randn(3, 1) * 1e1
state0 = state0.flatten()

prop = sorts.propagator.Kepler(
    settings=dict(
        in_frame="TEME",
        out_frame="ITRS",
    )
)

measurements = [(odlab.source_to_model(df, "radar_pair"), [df]) for df in dfs]

epoch = Time("2009-05-01T02:37:0", format="isot", scale="utc")

state_generator = methods.sortsPropagator(
    epoch, prop, propagator_args={"M0": pyorb.M_earth}
)

posterior = methods.posterior.GaussianError(
    measurements,
    state_generator,
)

logl = posterior.loglikelihood(state0_true.flatten())
print("log likelihood from posterior: ", logl)

deltas = np.array([1e1, 1e1, 1e1, 1e0, 1e0, 1e0], dtype=np.float64)
sigma_orb = posterior.linear_covariance_estimate(state0_true.flatten(), deltas)
print("Linearized posterior covariance estimate using estimated jacobian:")
print(sigma_orb)
print("Diagonal standard deviation: ", np.sqrt(np.diag(sigma_orb)))

solver = methods.solvers.ScipyMaximize(
    method = "Nelder-Mead",
    scipy_options = {},
)

result = solver.run(posterior, state0)

print(result)
