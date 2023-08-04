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
import odlab.methods.native as methods
import sorts
import pyorb

np.random.seed(897824)

data_dir = (pathlib.Path(__file__).parent / "tmp").resolve()

dfs = odlab.glob_sources(data_dir, {"radar_hdf": "sim_radar*.h5"})
assert len(dfs) > 0, "Run the 'simulate_radar_data.py' before trying this"

state0_true = np.load(data_dir / "state0.npy")
state0 = state0_true.copy()
state0[:3] += np.random.randn(3, 1)*1e3
state0[3:] += np.random.randn(3, 1)*1e1

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

od_solver = methods.MaximizeGaussianErrorPosterior(
    measurements, 
    state_generator,
)
