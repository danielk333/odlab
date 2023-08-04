#!/usr/bin/env python

"""Example showing how instrument models work

"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time

import odlab
import sorts
import pyorb


prop = sorts.propagator.Kepler(
    settings=dict(
        in_frame="ITRS",
        out_frame="ITRS",
    )
)

state0 = np.array(
    [-7100297.113, -3897715.442, 18568433.707, 86.771, -3407.231, 2961.571]
)
t = np.linspace(0, 600, num=100)
epoch = Time(53005, format="mjd", scale="utc")

# Generate some simulated states
states = prop.propagate(t, state0, epoch, M0=pyorb.M_earth)

print("Available models:", odlab.MODELS.keys())

print("Radar Pair")
print("input data : ", odlab.instrument_models.RadarPair.INPUT_DATA)
print("output data: ", odlab.instrument_models.RadarPair.OUTPUT_DATA)

ski_ecef = sorts.frames.geodetic_to_ITRS(69.34023844, 20.313166, 0.0, degrees=True)

# Instantiate the model
radar_model = odlab.instrument_models.RadarPair(
    {
        "tx_ecef": ski_ecef,
        "rx_ecef": ski_ecef,
    }
)
print(radar_model)
# Or by its name if that is preferred
cameara_model = odlab.get_model({"st_ecef": ski_ecef}, "camera")
print(cameara_model)

sim_radar_observations = radar_model.evaluate(t, states)
sim_camera_observations = cameara_model.evaluate(t, states)

fig, axes = plt.subplots(2, 2, figsize=(15, 15), sharex="all")

axes[0, 0].plot(t, sim_radar_observations["r"] * 1e-3)
axes[0, 0].set_ylabel('Radar two-way range [km]]')
axes[0, 1].plot(t, sim_radar_observations["v"] * 1e-3)
axes[0, 1].set_ylabel('Radar two-way velocity [km/s]]')

axes[1, 0].plot(t, sim_camera_observations["az"])
axes[1, 0].set_xlabel('Time [s]')
axes[1, 0].set_ylabel('Camera azimuth [deg]]')
axes[1, 1].plot(t, sim_camera_observations["el"])
axes[1, 1].set_xlabel('Time [s]')
axes[1, 1].set_ylabel('Camera elevation [deg]]')

plt.show()
