#!/usr/bin/env python

'''

'''

from pyod import RadarPair
from pyod import PropagatorOrekit
from pyod import OptimizeLeastSquares
from pyod import SourceCollection
from pyod import SourcePath
from pyod.plot import orbits
from pyod.datetime import mjd2npdt
from pyod.coordinates import geodetic2ecef

import pathlib
import numpy as np
import matplotlib.pyplot as plt

orekit_data = '/home/danielk/IRF/IRF_GITLAB/orekit_build/orekit-data-master.zip'

prop = PropagatorOrekit(
    orekit_data = orekit_data, 
    settings=dict(
        in_frame='ITRF',
        out_frame='ITRF',
        drag_force=False,
        radiation_pressure=False,
    )
)

state0 = np.array([-7100297.113,-3897715.442,18568433.707,86.771,-3407.231,2961.571])
t = np.linspace(0,1800,num=10)
mjd0 = 54952.08


ski_ecef = geodetic2ecef(69.34023844, 20.313166, 0.0)
data = dict(
    date = dates,
    date0 = mjd2npdt(mjd0),
    params = dict(
        A= 0.1, 
        m = 1.0,
    ),
    tx_ecef = ski_ecef,
    rx_ecef = ski_ecef,
)
radar = RadarPair(data, prop)
simulated_observation_data = radar.evaluate(state0)

data = [{
        'data': np.array([]),
        'meta': {},
        'index': 42,
    },
    {
        'data': np.array([]),
        'meta': {},
        'index': 43,
    }
]

paths = SourcePath.from_list(data, 'ram')

sources = SourceCollection(paths = paths)
sources.details()

exit()

variables = ['x', 'y', 'z', 'vx', 'vy', 'vz']
dtype = [(name, 'float64') for name in variables]
state0 = np.empty((1,), dtype=dtype)
for ind, name in enumerate(variables):
    state0[name] = state0_arr[ind]


input_data_state = {
    'sources': sources,
    'Model': RadarPair,
    'date0': mjd2npdt(mjd0),
    'params': {
        'm': 1.0,
        'A': 0.1,
    },
}

post = OptimizeLeastSquares(
    data = input_data_state,
    variables = variables,
    start = state0,
    prior = None,
    propagator = prop,
    method = 'Nelder-Mead',
    options = dict(
        maxiter = 10000,
        disp = True,
    ),
)

post.run()

print(post.results)

orbits(post)

plt.show()

