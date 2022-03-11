#!/usr/bin/env python

'''Example showing how Orekit propagator can be used

'''

import numpy as np
import matplotlib.pyplot as plt

import odlab
import sorts
import pyorb


prop = sorts.propagator.Kepler(
    settings=dict(
        in_frame='ITRS',
        out_frame='ITRS',
    )
)
print(prop)

state0 = np.array([-7100297.113, -3897715.442, 18568433.707,
                   86.771, -3407.231, 2961.571])
t = np.linspace(0, 0.2, num=100)
dates = odlab.times.mjd2npdt(53005 + t)

print(odlab.RadarPair.REQUIRED_DATA)
print(odlab.RadarPair.dtype)

ski_ecef = sorts.frames.geodetic_to_ITRS(69.34023844, 20.313166, 0.0)

data = dict(
    date = dates,
    date0 = odlab.times.mjd2npdt(53005),
    params = dict(
        A=1.0, 
        C_R = 1.0, 
        C_D = 1.0,
        m = 1,
    ),
    tx_ecef = ski_ecef,
    rx_ecef = ski_ecef,
)


radar = odlab.RadarPair(data, prop)

simulated_observation_data = radar.evaluate(state0, M0=pyorb.M_earth)


fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(211)
ax.plot(t, simulated_observation_data['r']*1e-3, '-b')

ax = fig.add_subplot(212)
ax.plot(t, simulated_observation_data['v']*1e-3, '-b')

plt.show()
