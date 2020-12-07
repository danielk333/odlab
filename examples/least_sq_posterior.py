#!/usr/bin/env python

'''

'''

import pyod
from pyod import RadarPair
from pyod import OptimizeLeastSquares
from pyod import SourceCollection
from pyod import SourcePath
from pyod.plot import orbits, residuals
from pyod.datetime import mjd2npdt
from pyod.coordinates import geodetic2ecef
from pyod.sources import RadarTracklet

import pathlib
import numpy as np
import matplotlib.pyplot as plt

# orekit_data = '/home/danielk/IRF/IRF_GITLAB/orekit_build/orekit-data-master.zip'
# prop = pyod.propagator.Orekit(
#     orekit_data = orekit_data, 
#     settings=dict(
#         in_frame='ITRS',
#         out_frame='ITRS',
#         drag_force=True,
#         radiation_pressure=False,
#     )
# )

np.random.seed(89782364)

#how to build a area-to-mass-ratio log10 based propagator
class MySGP4(pyod.propagator.SGP4):

    def propagate(self, t, state0, epoch, **kwargs):
        A_div_m = 10.0**kwargs['log10A_div_m']
        m = kwargs.pop('m', 1.0)
        A = A_div_m*m
        del kwargs['log10A_div_m']
        return super().propagate(t, state0, epoch=epoch, A=A, m=m, **kwargs)


prop = MySGP4(
    settings=dict(
        in_frame='ITRS',
        out_frame='ITRS',
    )
)

A = 2.0
m = 1.0

state0 = np.array([-7100297.113,-3897715.442,18568433.707,86.771,-3407.231,2961.571, np.log10(A/m)])
t = np.linspace(0,1800/(3600*24),num=10)
mjd0 = 54952.08
dates = mjd2npdt(mjd0 + t)
params = dict(C_D=2.3)

r_err = 1e3
v_err = 1e2

ski_ecef = geodetic2ecef(69.34023844, 20.313166, 0.0)
kar_ecef = geodetic2ecef(68.463862, 22.458859, 0.0)
kai_ecef = geodetic2ecef(68.148205, 19.769894, 0.0)

rx_list = [ski_ecef, kar_ecef, kai_ecef]

source_data = []

for rx in rx_list:
    data = dict(
        date = dates,
        date0 = mjd2npdt(mjd0),
        tx_ecef = ski_ecef,
        rx_ecef = rx,
    )
    radar = RadarPair(data, prop)
    sim_data = radar.evaluate(state0[:6], log10A_div_m=state0[6], **params)

    radar_data = np.empty((len(t),), dtype=RadarTracklet.dtype)
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


state_variables = ['x', 'y', 'z', 'vx', 'vy', 'vz']
variables = state_variables + ['log10A_div_m']
dtype = [(name, 'float64') for name in variables]
state0_named = np.empty((1,), dtype=dtype)
true_state = np.empty((1,), dtype=dtype)
start_err = [5e3]*3 + [1e2]*3 + [0.5]
for ind, name in enumerate(variables):
    state0_named[name] = state0[ind] + np.random.randn(1)*start_err[ind]
    true_state[name] = state0[ind]


input_data_state = {
    'sources': sources,
    'Models': [RadarPair]*len(sources),
    'date0': mjd2npdt(mjd0),
    'params': params,
}

post = OptimizeLeastSquares(
    data = input_data_state,
    variables = variables,
    state_variables = state_variables,
    start = state0_named,
    prior = None,
    propagator = prop,
    method = 'Nelder-Mead',
    options = dict(
        maxiter = 10000,
        disp = False,
        xatol = 1e-3,
    ),
)

deltas = [0.1]*3 + [0.05]*3 + [0.01]
data0, J, Sigma = post.model_jacobian(state0_named, deltas)

print(J)
print(Sigma)

post.run()

print(post.results)


print('Start error:')
for var in variables:
    if var in state_variables:
        print('{:<3}: {:.3f}'.format(var, (state0_named[var][0] - true_state[var][0])*1e-3))
    else:
        print('{:<3}: {:.3f}'.format(var, state0_named[var][0] - true_state[var][0]))


print('\n\nTrue error:')
for var in variables:
    if var in state_variables:
        print('{:<3}: {:.3f}'.format(var, (post.results.MAP[var][0] - true_state[var][0])*1e-3))
    else:
        print('{:<3}: {:.3f}'.format(var, post.results.MAP[var][0] - true_state[var][0]))

# orbits(post, true=true_state)
# residuals(post, [state0_named, true_state,post.results.MAP], ['Start', 'True', 'MAP'], ['-b', '-r', '-g'], absolute=False)
residuals(post, [state0_named, true_state,post.results.MAP], ['Start', 'True', 'MAP'], ['-b', '-r', '-g'], absolute=True)

plt.show()

