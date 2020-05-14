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
from .least_squares import OptimizeLeastSquares



class MCMCLeastSquares(OptimizeLeastSquares):
    '''Markov Chain Monte Carlo sampling of the posterior, assuming all measurement errors are Gaussian (thus the log likelihood becomes a least squares).
    '''

    REQUIRED = OptimizeLeastSquares.REQUIRED + [
        'steps',
        'step',
    ]

    OPTIONAL = {
        'method': 'SCAM',
        'method_options': {},
        'prior': None,
        'tune': 1000,
        'log_vars': [],
        'proposal': 'normal',
    }

    def __init__(self, data, variables, **kwargs):
        super(MCMCLeastSquares, self).__init__(data, variables, **kwargs)
    
    @mpi_wrap
    def run(self):
        if self.kwargs['start'] is None and self.kwargs['prior'] is None:
            raise ValueError('No start value or prior given.')
        
        start = self.kwargs['start']
        xnow = np.copy(start)
        step = np.copy(self.kwargs['step'])

        steps = self.kwargs['tune'] + self.kwargs['steps']
        chain = np.empty((steps,), dtype=start.dtype)
        
        logpost = self.evalute(xnow)

        if self.kwargs['method'] == 'SCAM':
            
            accept = np.zeros((len(self.variables),), dtype=start.dtype)
            tries = np.zeros((len(self.variables),), dtype=start.dtype)
            
            proposal_cov = np.eye(len(self.variables), dtype=np.float64)
            proposal_mu = np.zeros((len(self.variables,)), dtype=np.float64)

            print('\n{} running {}'.format(type(self).__name__, self.kwargs['method']))
            pbar = tqdm(range(steps), ncols=100)
            for ind in pbar:
                pbar.set_description('Sampling log-posterior = {:<10.3f} '.format(logpost))

                xtry = np.copy(xnow)

                pi = int(np.floor(np.random.rand(1)*len(self.variables)))
                var = self.variables[pi]

                proposal = np.random.multivariate_normal(proposal_mu, proposal_cov)
                
                vstep = proposal[pi]*step[0][var]

                if var in self.kwargs['log_vars']:
                    xtry[var] = 10.0**(np.log10(xtry[var]) + vstep)
                else:
                    xtry[var] += vstep
                
                logpost_try = self.evalute(xtry)
                alpha = np.log(np.random.rand(1))
                
                if logpost_try > logpost:
                    _accept = True
                elif (logpost_try - alpha) > logpost:
                    _accept = True
                else:
                    _accept = False
                
                tries[var] += 1.0
                
                if _accept:
                    logpost = logpost_try
                    xnow = xtry
                    accept[var] += 1.0

                if ind % 100 == 0 and ind > 0:
                    for name in self.variables:
                        ratio = accept[0][name]/tries[0][name]

                        if ratio > 0.5:
                            step[0][name] *= 2.0
                        elif ratio < 0.3:
                            step[0][name] /= 2.0
                        
                        accept[0][name] = 0.0
                        tries[0][name] = 0.0
                

                if ind % (steps//100) == 0 and ind > 0:
                    if self.kwargs['proposal'] == 'adaptive':
                        _data = np.empty((len(self.variables), ind), dtype=np.float64)
                        for dim, var in enumerate(self.variables):
                            _data[dim,:] = chain[:ind][var]
                        _proposal_cov = np.corrcoef(_data)
                        
                        if not np.any(np.isnan(_proposal_cov)):
                            proposal_cov = _proposal_cov


                chain[ind] = xnow
            
        chain = chain[self.kwargs['tune']:]

        self.results.trace = chain.copy()
        self._fill_results()

        return self.results


    def _fill_results(self):
        post_map = np.empty((1,), dtype=self.results.trace.dtype)
        for var in self.variables:
            post_map[var] = np.mean(self.results.trace[var])
        
        self.results.MAP = post_map
        self.loglikelihood(post_map)
        self.results.residuals = copy.deepcopy(self._tmp_residulas)
        self.results.date = self.data['date0']
    


