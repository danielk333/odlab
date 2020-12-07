#!/usr/bin/env python

'''

'''

#Python standard import
import copy

#Third party import
import numpy as np


#Local import
from . import datetime as datetime_local
from . import coordinates


class ForwardModel(object):

    dtype = [] #this is the dtype that is returned by the model

    REQUIRED_DATA = [
        'date',
        'date0',
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


    def get_states(self, state, **kw):
        states = self.propagator.propagate(
            self.data['t'],
            state,
            self.data['mjd0'],
            **kw
        )

        return states


    def evaluate(self, state, **kw):
        '''Evaluate forward model
        '''
        raise NotImplementedError()


    def distance(self, sim, obs):
        '''Calculates the distance between variable points

        Override this method to include non-standard distance measures 
        and coordinate transforms into posterior evaluation
        '''
        distances = np.empty(sim.shape, dtype=sim.dtype)
        for name in sim.dtype.names:
            distances[name] = obs[name] - sim[name]
        return distances



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


    def evaluate(self, state, **kw):
        '''Evaluate forward model
        '''

        states = self.get_states(state, **kw)

        sim_dat = np.empty((len(self.data['t']), ), dtype=RadarPair.dtype)

        for ind in range(len(self.data['t'])):
            r_obs, v_obs = RadarPair.generate_measurements(states[:,ind], self.data['rx_ecef'], self.data['tx_ecef'])
            sim_dat[ind]['r'] = r_obs
            sim_dat[ind]['v'] = v_obs

        return sim_dat



class CameraStation(ForwardModel):

    dtype = [
        ('az', 'float64'),
        ('el', 'float64'),
    ]
    
    REQUIRED_DATA = ForwardModel.REQUIRED_DATA + [
        'ecef',
    ]

    def __init__(self, data, propagator, **kwargs):
        super(CameraStation, self).__init__(data, propagator, **kwargs)


    @staticmethod
    def generate_measurements(state_ecef, ecef, lat, lon):

        x = state_ecef[:3] - ecef
        r = coordinates.ecef2local(lat, lon, 0.0, x[0], x[1], x[2])
        azel = coordinates.cart2azel(r)

        return azel[0], azel[1]


    def distance(self, sim, obs):
        '''Calculates the distances between angles, includes wrapping
        '''
        distances = np.empty(sim.shape, dtype=sim.dtype)
        
        daz = obs['az'] - sim['az']
        daz_tmp = np.mod(obs['az'] + 540.0, 360.0) - np.mod(sim['az'] + 540.0, 360.0)
        inds_ = np.abs(daz) > np.abs(daz_tmp)
        daz[inds_] = daz_tmp[inds_]
        distances['el'] = obs['el'] - sim['el']
        distances['az'] = daz

        return distances

        

    def evaluate(self, state, **kw):
        '''Evaluate forward model
        '''

        states = self.get_states(state, **kw)

        geo = coordinates.ecef2geodetic(self.data['ecef'][0], self.data['ecef'][1], self.data['ecef'][2])

        sim_dat = np.empty((len(self.data['t']), ), dtype=CameraStation.dtype)

        for ind in range(len(self.data['t'])):
            az_obs, el_obs = CameraStation.generate_measurements(states[:,ind], self.data['ecef'], geo[0], geo[1])
            sim_dat[ind]['az'] = az_obs
            sim_dat[ind]['el'] = el_obs

        return sim_dat




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


    def evaluate(self, state, **kw):
        '''Evaluate forward model
        '''

        states = self.get_states(state, **kw)

        sim_dat = np.empty((len(self.data['t']), ), dtype=EstimatedState.dtype)

        for ind in range(len(self.data['t'])):
            for dim, npd in enumerate(EstimatedState.dtype):
                name, _ = npd
                sim_dat[ind][name] = states[dim,ind]

        return sim_dat


