#!/usr/bin/env python

'''

'''

#Python standard import


#Third party import
import numpy as np


#Local import
from . import datetime as datetime_local


class ForwardModel(object):

    dtype = [] #this is the dtype that is returned by the model

    REQUIRED_DATA = [
        'date',
        'date0',
        'params',
    ]

    def __init__(self, data, propagator, coord='cart', **kwargs):
        for key in self.REQUIRED_DATA:
            if key not in data:
                raise ValueError('Data field {} is mandatory for {}'.format(key, type(self).__name__))
        
        self.data = data
        self.propagator = propagator
        self.coord = coord
        
        self.data['mjd0'] = datetime_local.npdt2mjd(self.data['date0'])
        t = (self.data['date'] - self.data['date0'])/np.timedelta64(1, 's')
        self.data['t'] = t

    def get_states(self, state):
        states = self.propagator.propagate(
            self.data['t'],
            state,
            self.data['mjd0'],
            **self.data['params']
        )

        return states

    def evaluate(self, state):
        '''Evaluate forward model
        '''
        raise NotImplementedError()



class RadarPair(ForwardModel):

    dtype = [
        ('r', 'float64'),
        ('v', 'float64'),
    ]
    
    REQUIRED_DATA = ForwardModel.REQUIRED_DATA + [
        'tx_ecef',
        'rx_ecef',
    ]

    def __init__(self, data, propagator, **kwargs):
        super(RadarPair, self).__init__(data, propagator, **kwargs)


    @staticmethod
    def generate_measurements(state_ecef, rx_ecef, tx_ecef):

        r_tx = tx_ecef - state_ecef[:3]
        r_rx = rx_ecef - state_ecef[:3]

        r_tx_n = np.linalg.norm(r_tx)
        r_rx_n = np.linalg.norm(r_rx)
        
        r_sim = r_tx_n + r_rx_n
        
        v_tx = -np.dot(r_tx, state_ecef[3:])/r_tx_n
        v_rx = -np.dot(r_rx, state_ecef[3:])/r_rx_n

        v_sim = v_tx + v_rx

        return r_sim, v_sim


    def evaluate(self, state):
        '''Evaluate forward model
        '''

        states = self.get_states(state)

        obs_dat = np.empty((len(self.data['t']), ), dtype=RadarPair.dtype)

        for ind in range(len(self.data['t'])):
            r_obs, v_obs = RadarPair.generate_measurements(states[:,ind], self.data['rx_ecef'], self.data['tx_ecef'])
            obs_dat[ind]['r'] = r_obs
            obs_dat[ind]['v'] = v_obs

        return obs_dat



class EstimatedState(ForwardModel):

    dtype = [
        ('x', 'float64'),
        ('y', 'float64'),
        ('z', 'float64'),
        ('vx', 'float64'),
        ('vy', 'float64'),
        ('vz', 'float64'),
    ]

    def __init__(self, data, propagator, **kwargs):
        super(EstimatedState, self).__init__(data, propagator, **kwargs)


    def evaluate(self, state):
        '''Evaluate forward model
        '''

        states = self.get_states(state)

        obs_dat = np.empty((len(self.data['t']), ), dtype=EstimatedState.dtype)

        for ind in range(len(self.data['t'])):
            for dim, npd in enumerate(EstimatedState.dtype):
                name, _ = npd
                obs_dat[ind][name] = states[dim,ind]

        return obs_dat

