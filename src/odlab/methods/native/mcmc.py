#!/usr/bin/env python

'''

'''

# Python standard import
import copy

# Third party import
from tqdm import tqdm
import numpy as np

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except ImportError:
    class COMM_WORLD:
        size = 1
        rank = 0
    comm = COMM_WORLD()

# Local import
from .least_squares import OptimizeLeastSquares


def mpi_wrap(run):

    def new_run(self, *args, **kwargs):

        run_mpi = self.kwargs.get('MPI', True)
        if run_mpi and comm.size > 1:
            steps = self.kwargs['steps']
            self.kwargs['steps'] = len(range(comm.rank, steps, comm.size))

            results0 = run(self, *args, **kwargs)
            trace0 = results0.trace

            if comm.rank == 0:
                trace = np.empty((0,), dtype=trace0.dtype)
                trace_pid = comm.gather(trace0, root=0)
                for pid in range(comm.size):
                    trace = np.append(trace, trace_pid[pid])
                    trace_pid[pid] = None
                del trace_pid

            if comm.rank != 0:
                del trace0
                del results0
                self.results = None
            else:
                self.results.trace = trace
                self._fill_results()

            return self.results
        else:
            return run(self, *args, **kwargs)

    return new_run


class MCMCLeastSquares(OptimizeLeastSquares):
    '''Markov Chain Monte Carlo sampling of the posterior, 
    assuming all measurement errors are Gaussian (thus the log likelihood 
    becomes a least squares).
    '''

    REQUIRED = OptimizeLeastSquares.REQUIRED + [
        'steps',
        'step',
    ]

    OPTIONAL = {
        'method': 'SCAM',
        'method_options': {
            'accept_max': 0.5,
            'accept_min': 0.3,
            'adapt_interval': 1000,
            'proposal_adapt_interval': 10000,
        },
        'prior': None,
        'tune': 1000,
        'proposal': 'normal',
        'jacobian_delta': 0.1,
        'MPI': True,
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

            if self.kwargs['proposal'] in ['normal', 'adaptive']:
                proposal_cov = np.eye(len(self.variables), dtype=np.float64)
                proposal_mu = np.zeros(
                    (len(self.variables,)), dtype=np.float64)
                proposal_axis = np.eye(len(self.variables), dtype=np.float64)
            elif self.kwargs['proposal'] == 'LinSigma':

                deltas = self.kwargs['jacobian_delta']
                if not isinstance(deltas, np.ndarray):
                    deltas = np.ones((len(self.variables),),
                                     dtype=np.float64)*deltas

                Sigma_orb = self.linear_MAP_covariance(
                    start, deltas, prior_cov_inv=None)

                proposal_mu = np.zeros(
                    (len(self.variables,)), dtype=np.float64)
                eigs, proposal_axis = np.linalg.eig(Sigma_orb)
                proposal_cov = np.diag(eigs)
            else:
                raise ValueError(f'proposal option "{self.kwargs["proposal"]}"\
                 not recognized')

            print('\n{} running {}'.format(
                type(self).__name__, self.kwargs['method']))
            pbars = []
            for pbar_id in range(comm.size):
                pbars.append(tqdm(range(steps), ncols=100))
            pbar = pbars[comm.rank]

            for ind in pbar:
                pbar.set_description(
                    'Sampling log-posterior = {:<10.3f} '.format(logpost))

                xtry = np.copy(xnow)

                pi = int(np.floor(np.random.rand(1)*len(self.variables)))
                var = self.variables[pi]

                proposal0 = np.random.multivariate_normal(
                    proposal_mu, proposal_cov)
                proposal = proposal0[pi]*proposal_axis[:, pi]

                for vind, v in enumerate(self.variables):
                    xtry[0][v] += proposal[vind]*step[0][var]

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

                ad_inv = self.kwargs['method_options']['adapt_interval']
                cov_ad_inv = self.kwargs['method_options']['proposal_adapt_interval']
                ac_min = self.kwargs['method_options']['accept_min']
                ac_max = self.kwargs['method_options']['accept_max']
                if ind % ad_inv == 0 and ind > 0:
                    for name in self.variables:
                        ratio = accept[0][name]/tries[0][name]

                        if ratio > ac_max:
                            step[0][name] *= 2.0
                        elif ratio < ac_min:
                            step[0][name] /= 2.0

                        accept[0][name] = 0.0
                        tries[0][name] = 0.0

                if ind % cov_ad_inv == 0 and ind > 0 and self.kwargs['proposal'] == 'adaptive':
                    _data = np.empty(
                        (len(self.variables), ind), 
                        dtype=np.float64,
                    )
                    for dim, var in enumerate(self.variables):
                        _data[dim, :] = chain[:ind][var]
                    _proposal_cov = np.corrcoef(_data)

                    if not np.any(np.isnan(_proposal_cov)):
                        eigs, proposal_axis = np.linalg.eig(_proposal_cov)
                        proposal_cov = np.diag(eigs)

                chain[ind] = xnow
        else:
            raise ValueError('No method found')

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
