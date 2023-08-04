#!/usr/bin/env python

'''

'''

import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time, TimeDelta

import odlab
import sorts
import pyorb


np.random.seed(89782364)

odlab.profiler.enable('odlab')

prop = sorts.propagator.Kepler(
    settings=dict(
        in_frame='ITRS',
        out_frame='ITRS',
    )
)


state0 = np.array([-7100297.113, -3897715.442, 18568433.707,
                   86.771, -3407.231, 2961.571])
t = np.linspace(0, 1800/(3600*24), num=10)
mjd0 = 54952.08
dates = Time(mjd0 + t, format="mjd", scale="utc")
params = dict(M0=pyorb.M_earth)

r_err = 1e3
v_err = 1e2

ski_ecef = sorts.frames.geodetic_to_ITRS(69.34023844, 20.313166, 0.0)
kar_ecef = sorts.frames.geodetic_to_ITRS(68.463862, 22.458859, 0.0)
kai_ecef = sorts.frames.geodetic_to_ITRS(68.148205, 19.769894, 0.0)

rx_list = [ski_ecef, kar_ecef, kai_ecef]

source_data = []

for rx in rx_list:
    data = dict(
        date = dates,
        date0 = odlab.times.mjd2npdt(mjd0),
        tx_ecef = ski_ecef,
        rx_ecef = rx,
    )
    radar = odlab.RadarPair(data, prop)
    sim_data = radar.evaluate(state0, **params)

    radar_data = np.empty((len(t),), dtype=odlab.sources.RadarTracklet.dtype)
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
    })

paths = odlab.SourcePath.from_list(source_data, 'ram')

sources = odlab.SourceCollection(paths = paths)
sources.details()


state_variables = ['x', 'y', 'z', 'vx', 'vy', 'vz']
variables = state_variables
dtype = [(name, 'float64') for name in variables]
state0_named = np.empty((1,), dtype=dtype)
true_state = np.empty((1,), dtype=dtype)
start_err = [0.5e3]*3 + [0.1e2]*3
for ind, name in enumerate(variables):
    state0_named[name] = state0[ind] + np.random.randn(1)*start_err[ind]
    true_state[name] = state0[ind]


input_data_state = {
    'sources': sources,
    'Models': [odlab.RadarPair]*len(sources),
    'date0': odlab.times.mjd2npdt(mjd0),
    'params': params,
}

post = odlab.OptimizeLeastSquares(
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

deltas = [0.1]*3 + [0.05]*3
Sigma_orb = post.linear_MAP_covariance(true_state, deltas)
print('Linearized MAP covariance:')
print(Sigma_orb)


post.run()

print(post.results)


print('Start error:')
for var in variables:
    if var in state_variables:
        print('{:<3}: {:.3f}'.format(
            var, (state0_named[var][0] - true_state[var][0])*1e-3))
    else:
        print('{:<3}: {:.3f}'.format(
            var, state0_named[var][0] - true_state[var][0]))


print('\n\nTrue error:')
for var in variables:
    if var in state_variables:
        print('{:<3}: {:.3f}'.format(
            var, (post.results.MAP[var][0] - true_state[var][0])*1e-3))
    else:
        print('{:<3}: {:.3f}'.format(
            var, post.results.MAP[var][0] - true_state[var][0]))

print(odlab.profiler)

# odlab.plot.orbits(post, true=true_state)
# odlab.plot.residuals(post, [state0_named, true_state, post.results.MAP], [
#           'Start', 'True', 'MAP'], ['-b', '-r', '-g'], absolute=False)
odlab.plot.residuals(post, [state0_named, true_state, post.results.MAP], [
          'Start', 'True', 'MAP'], ['o-b', '-r', 'x-g'], absolute=True)

plt.show()
