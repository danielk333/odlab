#!/usr/bin/env python

'''

'''

#Python standard import
import os
import copy

#Third party import
from tqdm import tqdm
import scipy.stats
import scipy.optimize as optimize
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
from mpi4py import MPI


comm = MPI.COMM_WORLD
#Local import
from .. import sources
from .posterior import Posterior
from .posterior import _named_to_enumerated, _enumerated_to_named


class OptimizeLeastSquares(Posterior):

    REQUIRED_DATA = [
        'sources',
        'Models',
        'date0',
        'params',
    ]

    REQUIRED = [
        'start',
        'propagator',
        'state_variables'
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

        st_var = self.kwargs['state_variables']
        self._variable_to_state = []
        self._param_to_state = []
        self._variable_to_param = []
        for din, sv in enumerate(st_var):
            try:
                vari = self.variables.index(sv)
                self._variable_to_state.append((din, vari))
            except ValueError:
                self._param_to_state.append((din, sv))
        for vari, var in enumerate(self.variables):
            if var not in st_var:
                self._variable_to_param.append((var, vari))


        for source_ind, source in enumerate(data['sources']):
            if not isinstance(source, sources.ObservationSource):
                raise ValueError('Non-observation data detected, "{}" not supported'.format(type(source)))

            Model__ = self.data['Models'][source_ind]

            req_args = copy.copy(Model__.REQUIRED_DATA)
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
                Model=Model__,
                **model_data
            )
            self._models.append(model)
        
        self._tmp_residulas = []
        for ind in range(len(self._models)):
            self._tmp_residulas.append(
                np.empty((len(self.data['sources'][ind].data),), dtype=self._models[ind].dtype)
            )
        

    def model_jacobian(self, state0, deltas):
        '''Calculate the observation and its numerical Jacobean of a state given the current models. 

        #TODO: Docstring
        '''

        tracklets = self.data['sources']
        n_sources = len(tracklets)

        state_, params = self._get_state_param(state0)

        meas_n = 0
        data0 = []
        for ind in range(n_sources):
            data0 += [self._models[ind].evaluate(state_, **params)]
            meas_n += len(self._models[ind].data['t'])*len(self._models[ind].dtype)

        
        Sigma = np.zeros([meas_n,], dtype=np.float64)

        meas_ind0 = 0
        for sc_ind in range(n_sources):
            tn = len(self._models[sc_ind].data['t'])
            var_ind = 0
            for var, dt in self._models[sc_ind].dtype:

                start_ind = meas_ind0 + var_ind*tn
                end_ind = meas_ind0 + (var_ind + 1)*tn
                Sigma[start_ind:end_ind] = tracklets[sc_ind].data[var + '_sd']**2.0

                var_ind += 1
            meas_ind0 += tn*len(self._models[sc_ind].dtype)


        J = np.zeros([meas_n,len(self.variables)], dtype=np.float64)

        for ind, var in enumerate(self.variables):

            dstate = np.copy(state0)
            dstate[var][0] = dstate[var][0] + deltas[ind]

            dstate_, dparams = self._get_state_param(dstate)

            meas_ind0 = 0
            for sc_ind in range(n_sources):
                ddata = self._models[sc_ind].evaluate(dstate_, **dparams)

                tn = len(self._models[sc_ind].data['t'])
                var_ind = 0
                for var, dt in self._models[sc_ind].dtype:
                    dvar = (ddata[var] - data0[sc_ind][var])/deltas[ind]

                    start_ind = meas_ind0 + var_ind*tn
                    end_ind = meas_ind0 + (var_ind + 1)*tn
                    J[start_ind:end_ind, ind] = dvar

                    var_ind += 1

                meas_ind0 += tn*len(self._models[sc_ind].dtype)

        return data0, J, Sigma


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


    def _get_state_param(self, state):
        state_all = _named_to_enumerated(state, self.variables)

        st_var = self.kwargs['state_variables']

        state_ = np.empty((len(st_var),), dtype=np.float64)
        params = copy.copy(self.data['params'])

        for sti, vari in self._variable_to_state:
            state_[sti] = state_all[vari]

        for sti, var in self._param_to_state:
            state_[sti] = self.data['params'][var]

        for par, vari in self._variable_to_param:
            params[par] = state_all[vari]

        return state_, params


    def loglikelihood(self, state):
        '''The loglikelihood function
        '''

        tracklets = self.data['sources']
        n_tracklets = len(tracklets)
        
        state_, params = self._get_state_param(state)

        logsum = 0.0
        for ind in range(n_tracklets):
            
            sim_data = self._models[ind].evaluate(state_, **params)
            _residuals = self._models[ind].distance(sim_data, tracklets[ind].data)
            for name, _ in self._models[ind].dtype:
                self._tmp_residulas[ind][name] = _residuals[name]

            num = len(self._models[ind].data['t'])
            
            if 'cov' in tracklets[ind].data.dtype.names:
                names = tracklets[ind].meta['variables']

                for ti in range(num):
                    xi = structured_to_unstructured(_residuals[ti][names])
                    cov = tracklets[ind].data['cov'][ti]
                    
                    logsum += xi.T*np.linalg.inv(cov)*xi
            else:
                for name, _ in self._models[ind].dtype:
                    logsum += np.sum(-1.0*_residuals[name]**2.0/(tracklets[ind].data[name + '_sd']**2.0))

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

        pbars = []
        for pbar_id in range(comm.size):
            pbars.append(tqdm(total=maxiter, ncols=100))
        pbar = pbars[comm.rank]
        for ind in range(comm.size):
            if ind != comm.rank: pbars[ind].close()

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

