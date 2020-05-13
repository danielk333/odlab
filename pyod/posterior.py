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
from . import sources


def _named_to_enumerated(state, names):
    return np.array([state[name] for name in names], dtype=np.float64).flatten()


def _enumerated_to_named(state, names):
    _dtype = [(name, 'float64') for name in names]
    _state = np.empty((1,), dtype=_dtype)
    for ind, name in enumerate(names):
        _state[name] = state[ind]
    return _state



class PosteriorParameters(object):

    attrs = [
        'variables',
        'trace',
        'MAP',
        'residuals',
        'date',
    ]

    def __init__(self, **kwargs):
        for key in self.attrs:
            setattr(self, key, kwargs.get(key, None))

    @classmethod
    def load_h5(cls, path):
        '''Load evaluated posterior
        '''
        results = cls()
        if isinstance(path, sources.SourcePath):
            if path.ptype != 'file':
                raise TypeError('Can only load posterior data from file, not "{}"'.format(path.ptype))
            _path = path.data
        else:
            _path = path

        with h5py.File(_path, 'r') as hf:
            results.variables = hf.attrs['variables'].tolist()
            results.date = dpt.mjd2npdt(hf.attrs['date'])
            results.trace = hf['trace'].value
            results.MAP = hf['MAP'].value
            results.residuals = []
            grp = hf['residuals/']
            for key in grp:
                results.residuals.append(grp[key].value)

        return results


    def __getitem__(self, key):
        if key in self.trace.dtype.names:
            return self.trace[key]
        else:
            return KeyError('No results exists for "{}"'.format(key))


    def load(self, path):
        '''Load evaluated posterior
        '''
        if isinstance(path, sources.SourcePath):
            if path.ptype != 'file':
                raise TypeError('Can only load posterior data from file, not "{}"'.format(path.ptype))
            _path = path.data
        else:
            _path = path

        with h5py.File(_path, 'r') as hf:
            _vars = hf.attrs['variables'].tolist()
            if self.variables is not None:
                for var in _vars:
                    if var not in self.variables:
                        raise Exception('Variable spaces do not match between current and loaded data')
                for var in self.variables:
                    if var not in _vars:
                        raise Exception('Variable spaces do not match between current and loaded data')
            else:
                self.variables = _vars
            if self.results is not None:
                self.results = np.append(self.results, hf['results'].value)
            else:
                self.results = hf['results']
            if self.MAP is not None:
                self.MAP = np.append(self.MAP, hf['MAP'].value)
            else:
                self.MAP = hf['MAP']
            if self.residuals is not None:
                grp = hf['residuals/']
                for key in grp:
                    results.residuals.append(grp['{}'.format(ind)].value)
            else:
                self.residuals = []
                grp = hf['residuals/']
                for key in grp:
                    self.residuals.append(grp['{}'.format(ind)].value)
            if self.date is not None:
                if self.date != dpt.mjd2npdt(hf.attrs['date']):
                    raise Exception('Cannot load data from another epoch "{}" vs "{}"'.format(self.date, dpt.mjd2npdt(hf.attrs['date'])))



    def save(self, path):
        '''Save evaluated posterior
        '''
        if isinstance(path, sources.SourcePath):
            if path.ptype != 'file':
                raise TypeError('Can only write posterior data to file, not "{}"'.format(path.ptype))
            _path = path.data
        else:
            _path = path

        with h5py.File(_path, 'w') as hf:
            hf.attrs['variables'] = self.variables
            hf['trace'] = self.trace
            hf['MAP'] = self.MAP
            hf.attrs['date'] = dpt.npdt2mjd(self.date)

            grp = hf.create_group("residuals")
            for ind, resid in enumerate(self.residuals):
                grp.create_dataset(
                        '{}'.format(ind),
                        data=self.residuals[ind],
                    )


    def autocovariance(self, max_k = None, min_k = None):
        if max_k is None:
            max_k = len(self.trace)
        else:
            if max_k >= len(self.trace):
                max_k = len(self.trace)

        if min_k is None:
            min_k = 0
        else:
            if min_k >= len(self.trace):
                min_k = len(self.trace)-1


        gamma = np.empty((max_k-min_k,), dtype=self.trace.dtype)

        _n = len(self.trace)

        for var in self.variables:
            for k in range(min_k, max_k):
                covi = self.trace[var][:(_n-k)] - self.MAP[0][var]
                covik = self.trace[var][k:_n] - self.MAP[0][var]
                gamma[var][k] = np.sum( covi*covik )/float(_n)

        return gamma


    def batch_mean(self, batch_size):
        if batch_size > len(self.trace):
            raise Exception('Not enough samples to calculate batch statistics')

        _max = batch_size
        batches = len(self.trace)//batch_size
        batch_mean = np.empty((batches,), dtype=self.trace.dtype)
        for ind in range(batches):
            batch = self.trace[(_max - batch_size):_max]
            _max += batch_size

            for var in self.variables:
                batch_mean[ind][var] = np.mean(batch[var])

        return batch_mean


    def batch_covariance(self, batch_size):
        if batch_size > len(self.trace):
            raise Exception('Not enough samples to calculate batch statistics')

        batch_mean = self.batch_mean(batch_size)

        _max_str = int(np.max([len(var) for var in self.variables]))

        _dtype = self.trace.dtype.names
        _dtype = [('variable', 'U{}'.format(_max_str))] + [(name, 'float64') for name in _dtype]
        cov = np.empty((len(self.variables),), dtype=_dtype)
        for ind, xvar in enumerate(self.variables):
            for yvar in self.variables:
                cov[ind]['variable'] = xvar
                cov[ind][yvar] = np.mean( (batch_mean[xvar] - self.MAP[xvar])*(batch_mean[yvar] - self.MAP[yvar]) )/float(len(batch_mean))

        return cov


    def batch_variance(self, batch_size):
        if batch_size > len(self.trace):
            raise Exception('Not enough samples to calculate batch statistics')

        batch_mean = self.batch_mean(batch_size)

        variance = np.empty((1,), dtype=self.trace.dtype)
        for var in self.variables:
            variance[var] = np.mean( (batch_mean[var] - self.MAP[var])**2)

        return variance/float(len(batch_mean))


    def covariance_mat(self, variables=None):
        if variables is None:
            variables = self.variables

        cov = np.empty((len(variables),len(variables)), dtype=np.float64)

        mean = np.empty((1,), dtype=self.trace.dtype)
        for ind, xvar in enumerate(variables):
            mean[xvar] = np.mean(self.trace[xvar])

        for xind, xvar in enumerate(variables):
            for yind, yvar in enumerate(variables):
                cov[xind, yind] = np.sum( (self.trace[xvar] - mean[xvar])*(self.trace[yvar] - mean[yvar]) )/float(len(self.trace)-1)
        return cov


    def covariance(self):

        _max_str = int(np.max([len(var) for var in self.variables]))

        _dtype = self.trace.dtype.names
        _dtype = [('variable', 'U{}'.format(_max_str))] + [(name, 'float64') for name in _dtype]
        cov = np.empty((len(self.variables),), dtype=_dtype)

        mean = np.empty((1,), dtype=self.trace.dtype)
        for ind, xvar in enumerate(self.variables):
            mean[xvar] = np.mean(self.trace[xvar])

        for ind, xvar in enumerate(self.variables):
            for yvar in self.variables:
                cov[ind]['variable'] = xvar
                cov[ind][yvar] = np.sum( (self.trace[xvar] - mean[xvar])*(self.trace[yvar] - mean[yvar]) )/float(len(self.trace)-1)
        return cov

    def __str__(self):
        _str = ''
        _str += '='*10 + ' MAP state ' + '='*10 + '\n'
        for ind, var in enumerate(self.variables):
            _str += '{}: {}\n'.format(var, self.MAP[var])

        _str += '='*10 + ' Residuals ' + '='*10 + '\n'
        for ind, res in enumerate(self.residuals):
            _str += ' - Model {}'.format(ind) + '\n'
            for key in res.dtype.names:
                _str += ' -- mean({}) = {}'.format(key, np.mean(res[key])) + '\n'
        return _str



class Posterior(object):

    REQUIRED_DATA = []
    REQUIRED = []
    OPTIONAL = {}

    def __init__(self, data, variables, **kwargs):
        self.kwargs = self.OPTIONAL.copy()
        self.kwargs.update(kwargs)

        self.variables = variables
        self.data = data

        self.results = PosteriorParameters(variables = variables)


    def logprior(self, state):
        '''The logprior function, defaults to uniform if not implemented
        '''
        return 0.0


    def loglikelihood(self, state):
        '''The loglikelihood function
        '''
        raise NotImplementedError()

    
    def evalute(self, state):
        return self.logprior(state) + self.loglikelihood(state)


    def run(self):
        '''Evaluate posterior
        '''
        raise NotImplementedError()


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
        super(OptimizeLeastSquares, self).__init__(data, variables, **kwargs)

        for key in self.REQUIRED:
            if key not in kwargs:
                raise ValueError('Argument "{}" is mandatory'.format(key))

        for key in self.REQUIRED_DATA:
            if key not in data:
                raise ValueError('Data field "{}" is mandatory'.format(key))

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
                
                logsum -= np.sum(_residuals**2.0/(tracklets[ind].data[name + '_sd']**2.0))
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

