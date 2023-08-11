#!/usr/bin/env python

"""

Run the "simulate_radar_data.py" before trying this

"""
import pathlib

import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
import pandas as pd

import odlab
import odlab.methods as methods
import sorts
import pyorb

np.random.seed(897824)

data_dir = (pathlib.Path(__file__).parent / "tmp").resolve()

dfs = odlab.glob_sources(data_dir, {"radar_hdf": "sim_radar*.h5"})
assert len(dfs) > 0, "Run the 'simulate_radar_data.py' before trying this"

state0_true = np.load(data_dir / "state0.npy")
state0 = state0_true.copy()
state0[:3, :] += np.random.randn(3, 1)*1e3
state0[3:, :] += np.random.randn(3, 1)*1e1
state0 = state0.flatten()

prop = sorts.propagator.Kepler(
    settings=dict(
        in_frame="TEME",
        out_frame="ITRS",
    )
)

measurements = [
    (odlab.source_to_model(df, "radar_pair"), [df])
    for df in dfs
]

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

max_solver = methods.solvers.ScipyMaximize(
    method = "Nelder-Mead",
    scipy_options = {},
)

peak_result = max_solver.run(posterior, state0)

print(peak_result)

start1 = peak_result.x

step_size = deltas*0.1
steps = 10000

dist_solver = methods.solvers.Scam(
    step_size,
    proposal = 'LinSigma',
    accept_max = 0.5,
    accept_min = 0.3,
    adapt_interval = 500,
    tune = 1000,
)

chain = dist_solver.run(posterior, start1, steps)

df_chain = pd.DataFrame(chain.T, columns=["x", "y", "z", "vx", "vy", "vz"])
axes = pd.plotting.scatter_matrix(df_chain, figsize=(12, 8), s=1)

plt.show()

# odlab.plot.orbits(post, true=true_state)

# odlab.plot.residuals(
#     post, 
#     [state0_named, true_state, results.MAP], 
#     ['Start', 'True', 'MAP'], 
#     ['-b', '-r', '-g'], 
#     absolute=False,
# )
# odlab.plot.residuals(
#     post, 
#     [state0_named, true_state, results.MAP], 
#     ['Start', 'True', 'MAP'], 
#     ['-b', '-r', '-g'], 
#     absolute=True,
# )

# plt.show()

# exit()


# print(results)

# print('True error:')
# for var in variables:
# print(f'{var:<3}: {(results.MAP[var][0] - true_state[var][0])*1e-3:.3f}')

# odlab.plot.autocorrelation(results, max_k=steps)
# odlab.plot.trace(results, reference=true_state)
# odlab.plot.trace(results)
# odlab.plot.scatter_trace(results, reference=true_state)
