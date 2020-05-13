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


data_dir = '.' / pathlib.Path(__file__).parents[0] / 'example_data'

paths = SourcePath.recursive_folder(data_dir, ['tdm'])
sources = SourceCollection(paths = paths)
sources.details()

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

variables = ['x', 'y', 'z', 'vx', 'vy', 'vz']

dtype = [(name, 'float64') for name in variables]


state0_arr = np.array([
    -7100297.113,
    -3897715.442,
    18568433.707,
    86.771,
    -3407.231,
    2961.571,
])
# state0_arr = np.empty((6,), dtype=np.float64)
# state0_arr[:3] = geodetic2ecef(69.34023844, 20.313166, 0.0) + 10000
# state0_arr[3:] = 0.0
state0 = np.empty((1,), dtype=dtype)
for ind, name in enumerate(variables):
    state0[name] = state0_arr[ind]


input_data_state = {
    'sources': sources,
    'Model': RadarPair,
    'date0': mjd2npdt(54952.08),
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

