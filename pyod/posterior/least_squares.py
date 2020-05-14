#!/usr/bin/env python

'''

'''

#Python standard import
import os
import copy
import glob

#Third party import
import h5py
from tqdm import tqdm
import scipy
import scipy.stats
import scipy.optimize as optimize
import numpy as np

#Local import
from .. import sources
from .posterior import Posterior
from .posterior import _named_to_enumerated, _enumerated_to_named


class OptimizeLeastSquares(Posterior):

    REQUIRED_DATA = [
        'sources',
        'Model',
        'date0',
        'params',
    ]

    REQUIRED = [
        'start',
        'propagator',
    ]

    OPTIONAL = {
        'method': 'Nelder-Mead',
        'prior': None,
        'options': {},
    }

    def __init__(self, data, variables, **kwargs):
        for key in self.REQUIRED:
            if key not in kwargs:
                raise ValueError('Argument "{}" is mandatory'.format(key))

        for key in self.REQUIRED_DATA:
            if key not in data:
                raise ValueError('Data field "{}" is mandatory'.format(key))

        super(OptimizeLeastSquares, self).__init__(data, variables, **kwargs)

        self._models = []

        for source in data['sources']:
            if not isinstance(source, sources.ObservationSource):
                raise ValueError('Non-observation data detected, "{}" not supported'.format(type(source)))

            req_args = copy.copy(self.data['Model'].REQUIRED_DATA)
            for arg in source.avalible_data():
                if arg in req_args:
                    req_args.remove(arg)

            model_data = {
                'propagator': self.kwargs['propagator'],
            }
            for arg in req_args:
                if arg not in self.data:
                    raise TypeError('Model REQUIRED data "{}" not found'.format(arg))
                model_data[arg] = self.data[arg]

            model = source.generate_model(
                Model=self.data['Model'],
                **model_data
            )
            self._models.append(model)
        
        self._tmp_residulas = []
        for ind in range(len(self._models)):
            self._tmp_residulas.append(
                np.empty((len(self.data['sources'][ind].data),), dtype=self._models[ind].dtype)
            )
        

    def logprior(self, state):
        '''The logprior function
        '''
        logprob = 0.0
        
        if self.kwargs['prior'] is None:
            return logprob
        
        for prior in self.kwargs['prior']:
            _state = _named_to_enumerated(state, prior['variables'])
            dist = getattr(scipy.stats, prior['distribution'])
            
            _pr = dist.logpdf(_state, **prior['params'])
            if isinstance(_pr, np.ndarray):
                _pr = _pr[0]
            logprob += _pr
        
        return logprob



    def loglikelihood(self, state):
        '''The loglikelihood function
        '''

        tracklets = self.data['sources']
        n_tracklets = len(tracklets)
        state_ = _named_to_enumerated(state, self.variables)

        logsum = 0.0
        for ind in range(n_tracklets):
            
            sim_data = self._models[ind].evaluate(state_)

            for name, nptype in self._models[ind].dtype:
                _residuals = tracklets[ind].data[name] - sim_data[name]

                self._tmp_residulas[ind][name] = _residuals
                
                logsum += np.sum(-1.0*_residuals**2.0/(tracklets[ind].data[name + '_sd']**2.0))
        return 0.5*logsum


    def run(self):
        if self.kwargs['start'] is None and self.kwargs['prior'] is None:
            raise ValueError('No start value or prior given.')

        start = _named_to_enumerated(self.kwargs['start'], self.variables)

        maxiter = self.kwargs['options'].get('maxiter', 3000)

        def fun(x):
            _x = _enumerated_to_named(x, self.variables)

            try:
                val = self.evalute(_x)
            except:
                val = -np.inf
                raise
            
            pbar.update(1)
            pbar.set_description("Least Squares = {:<10.3f} ".format(-val))

            return -val
        
        print('\n{} running {}'.format(type(self).__name__, self.kwargs['method']))

        pbar = tqdm(total=maxiter, ncols=100)
        xhat = optimize.minimize(
            fun,
            start,
            method = self.kwargs['method'],
            options = self.kwargs['options'],
        )
        pbar.close()

        self.results.trace = _enumerated_to_named(xhat.x, self.variables)
        self.results.MAP = self.results.trace.copy()
        self.loglikelihood(self.results.MAP)
        self.results.residuals = copy.deepcopy(self._tmp_residulas)
        self.results.date = self.data['date0']

        return self.results


    def residuals(self, state):

        self.loglikelihood(state)

        residuals = []
        for ind, resid in enumerate(self._tmp_residulas):
            residuals.append({
                'date': self._models[ind].data['date'],
                'residuals': resid.copy(),
            })
        return residuals

