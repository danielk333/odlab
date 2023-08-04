#!/usr/bin/env python

'''

'''
import pathlib
import numpy as np
from astropy.time import Time
import pandas as pd

import odlab
import sorts
import pyorb

np.random.seed(89782364)

data_dir = (pathlib.Path(__file__).parent / 'tmp').resolve()

prop = sorts.propagator.Kepler(
    settings=dict(
        in_frame='ITRS',
        out_frame='ITRS',
    )
)

state0 = np.array([-7100297.113, -3897715.442, 18568433.707,
                   86.771, -3407.231, 2961.571])
t = np.linspace(0, 1800.0, num=10)
mjd0 = 54952.08
dates = Time(mjd0 + t/(3600*24), format="mjd", scale="utc")
params = dict(M0=pyorb.M_earth)

states = prop.propagate(t, state0, dates[0], M0=pyorb.M_earth)

r_err = 1e3
v_err = 1e2

ski_ecef = sorts.frames.geodetic_to_ITRS(69.34023844, 20.313166, 0.0)
kar_ecef = sorts.frames.geodetic_to_ITRS(68.463862, 22.458859, 0.0)
kai_ecef = sorts.frames.geodetic_to_ITRS(68.148205, 19.769894, 0.0)

rx_list = [ski_ecef, kar_ecef, kai_ecef]

for ind, rx_ecef in enumerate(rx_list):
    path = data_dir / f"sim_radar_{ind}.h5"
    meta = {
        "tx_ecef": ski_ecef,
        "rx_ecef": rx_ecef,
    }
    radar_model = odlab.instrument_models.RadarPair(meta)
    sim_data = radar_model.evaluate(t, states)

    radar_data = {}
    radar_data['date'] = dates.datetime64
    radar_data['r'] = sim_data['r'] + np.random.randn(len(t))*r_err
    radar_data['v'] = sim_data['v'] + np.random.randn(len(t))*v_err
    radar_data['r_sd'] = np.full((len(t),), r_err, dtype=np.float64)
    radar_data['v_sd'] = np.full((len(t),), v_err, dtype=np.float64)

    df = pd.DataFrame(radar_data)
    df.attrs.update(meta)

    print(f"Writing file {path}...")
    odlab.data.hdf.save_radar_hdfs(path, df)
