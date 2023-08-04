#!/usr/bin/env python

"""

"""
import pathlib
import numpy as np
from astropy.time import Time, TimeDelta
import pandas as pd

import odlab
import sorts
import pyorb

np.random.seed(89782364)

data_dir = (pathlib.Path(__file__).parent / "tmp").resolve()

prop = sorts.propagator.Kepler(
    settings=dict(
        in_frame="TEME",
        out_frame="ITRS",
    )
)

ski_ecef = sorts.frames.geodetic_to_ITRS(69.34023844, 20.313166, 0.0)
kar_ecef = sorts.frames.geodetic_to_ITRS(68.463862, 22.458859, 0.0)
kai_ecef = sorts.frames.geodetic_to_ITRS(68.148205, 19.769894, 0.0)

t = np.linspace(0, 180.0, num=20)
epoch = Time("2009-05-01T02:37:0", format="isot", scale="utc")
dates = epoch + TimeDelta(t, format="sec")
params = dict(M0=pyorb.M_earth)

r_ecef = ski_ecef + ski_ecef / np.linalg.norm(ski_ecef) * 500e3

r_teme = sorts.frames.convert(
    epoch,
    np.hstack([r_ecef, np.ones_like(r_ecef)]),
    in_frame="ITRS",
    out_frame="TEME",
)
r_teme = r_teme[:3]

orb = pyorb.Orbit(
    M0=pyorb.M_earth,
    m=0,
    num=1,
    epoch=epoch,
    degrees=True,
)
orb.update(
    a=np.linalg.norm(r_teme),
    e=0,
    omega=0,
    i=0,
    Omega=0,
    anom=0,
)
v_norm = orb.speed[0]

x_hat = np.array([1, 0, 0], dtype=np.float64)  # Z-axis unit vector
z_hat = np.array([0, 0, 1], dtype=np.float64)  # Z-axis unit vector
# formula for creating the velocity vectors
b3 = r_teme / np.linalg.norm(r_teme)  # r unit vector
b3 = b3.flatten()
b1 = np.cross(b3, z_hat)  # Az unit vector
if np.linalg.norm(b1) < 1e-12:
    b1 = np.cross(b3, x_hat)  # Az unit vector
b1 = b1 / np.linalg.norm(b1)
v_temes = v_norm * b1
orb.update(
    x=r_teme[0],
    y=r_teme[1],
    z=r_teme[2],
    vx=v_temes[0],
    vy=v_temes[1],
    vz=v_temes[2],
)

state0 = orb.cartesian
np.save(data_dir / "state0.npy", state0)

states = prop.propagate(t, state0, epoch, M0=pyorb.M_earth)

r_err = 1e3
v_err = 1e2

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
    radar_data["date"] = dates.datetime64
    radar_data["r"] = sim_data["r"] + np.random.randn(len(t)) * r_err
    radar_data["v"] = sim_data["v"] + np.random.randn(len(t)) * v_err
    radar_data["r_sd"] = np.full((len(t),), r_err, dtype=np.float64)
    radar_data["v_sd"] = np.full((len(t),), v_err, dtype=np.float64)

    df = pd.DataFrame(radar_data)
    df.attrs.update(meta)

    print(f"Writing file {path}...")
    odlab.data.hdf.save_radar_hdfs(path, df)
