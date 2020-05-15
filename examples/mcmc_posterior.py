#!/usr/bin/env python

'''

'''

from pyod import RadarPair
from pyod import PropagatorOrekit
from pyod import MCMCLeastSquares, OptimizeLeastSquares
from pyod import SourceCollection
from pyod import SourcePath
import pyod.plot as plots
from pyod.datetime import mjd2npdt
from pyod.coordinates import geodetic2ecef
from pyod.sources import TrackletSource

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
t = np.linspace(0,1800/(3600*24),num=20)
mjd0 = 54952.08
dates = mjd2npdt(mjd0 + t)
params = dict(A= 0.1, m = 1.0)

r_err = 1e3
v_err = 1e2

ski_ecef = geodetic2ecef(69.34023844, 20.313166, 0.0)
kar_ecef = geodetic2ecef(68.463862, 22.458859, 0.0)
kai_ecef = geodetic2ecef(68.148205, 19.769894, 0.0)

rx_list = [ski_ecef, kar_ecef, kai_ecef]
# rx_list = [ski_ecef]

source_data = []

for rx in rx_list:
    data = dict(
        date = dates,
        date0 = mjd2npdt(mjd0),
        params = params,
        tx_ecef = ski_ecef,
        rx_ecef = rx,
    )
    radar = RadarPair(data, prop)
    sim_data = radar.evaluate(state0)

    radar_data = np.empty((len(t),), dtype=TrackletSource.dtype)
    radar_data['date'] = dates
    radar_data['r'] = sim_data['r'] + np.random.randn(len(t))*r_err
    radar_data['v'] = sim_data['v'] + np.random.randn(len(t))*v_err
    radar_data['r_sd'] = np.full((len(t),), r_err, dtype=np.float64)
    radar_data['v_sd'] = np.full((len(t),), v_err, dtype=np.float64)

    source_data.append({
            'data': radar_data,
            'meta': dict(
                tx_ecef = ski_ecef,
                rx_ecef = rx,
            ),
            'index': 1,
        }
    )

paths = SourcePath.from_list(source_data, 'ram')

sources = SourceCollection(paths = paths)
sources.details()

variables = ['x', 'y', 'z', 'vx', 'vy', 'vz']
dtype = [(name, 'float64') for name in variables]
state0_named = np.empty((1,), dtype=dtype)
true_state = np.empty((1,), dtype=dtype)
start_err = [5e3]*3 + [1e2]*3

step_arr = np.array([1e2,1e2,1e2,1e1,1e1,1e1], dtype=np.float64)*10
step = np.empty((1,), dtype=dtype)
for ind, name in enumerate(variables):
    state0_named[name] = state0[ind] + np.random.randn(1)*start_err[ind]
    true_state[name] = state0[ind]
    step[name] = step_arr[ind]


input_data_state = {
    'sources': sources,
    'Model': RadarPair,
    'date0': mjd2npdt(mjd0),
    'params': params,
}

post_init = OptimizeLeastSquares(
    data = input_data_state,
    variables = variables,
    start = state0_named,
    prior = None,
    propagator = prop,
    method = 'Nelder-Mead',
    options = dict(
        maxiter = 10000,
        disp = True,
        xatol = 1e1,
    ),
)

post_init.run()

post = MCMCLeastSquares(
    data = input_data_state,
    variables = variables,
    start = post_init.results.MAP,
    prior = None,
    propagator = prop,
    method = 'SCAM',
    method_options = dict(
        accept_max = 0.6,
        accept_min = 0.2,
        adapt_interval = 500,
    ),
    steps = int(1.5e4),
    step = step,
    tune = 0,
)

post.run()

print(post.results)

print('True error:')
for var in variables:
    print('{:<3}: {:.3f}'.format(var, (post.results.MAP[var][0] - true_state[var][0])*1e-3))


plots.autocorrelation(post.results, max_k=1000)

plt.show()

plots.trace(post.results)
plots.scatter_trace(post.results)

plots.orbits(post, true=true_state)

plots.residuals(post, [state0_named, true_state,post.results.MAP], ['Start', 'True', 'MAP'], ['-b', '-r', '-g'], absolute=False)
plots.residuals(post, [state0_named, true_state,post.results.MAP], ['Start', 'True', 'MAP'], ['-b', '-r', '-g'], absolute=True)

plt.show()

