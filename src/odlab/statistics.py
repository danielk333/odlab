
def autocovariance(trace, max_k = None, min_k = None):
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
            gamma[var][k] = np.sum(covi*covik)/float(_n)

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
    _dtype = [('variable', 'U{}'.format(_max_str))] + \
        [(name, 'float64') for name in _dtype]
    cov = np.empty((len(self.variables),), dtype=_dtype)
    for ind, xvar in enumerate(self.variables):
        for yvar in self.variables:
            cov[ind]['variable'] = xvar
            cov[ind][yvar] = np.mean((batch_mean[xvar] - self.MAP[xvar])*(
                batch_mean[yvar] - self.MAP[yvar]))/float(len(batch_mean))

    return cov

def batch_variance(self, batch_size):
    if batch_size > len(self.trace):
        raise Exception('Not enough samples to calculate batch statistics')

    batch_mean = self.batch_mean(batch_size)

    variance = np.empty((1,), dtype=self.trace.dtype)
    for var in self.variables:
        variance[var] = np.mean((batch_mean[var] - self.MAP[var])**2)

    return variance/float(len(batch_mean))

def covariance_mat(self, variables=None):
    if variables is None:
        variables = self.variables

    cov = np.empty((len(variables), len(variables)), dtype=np.float64)

    mean = np.empty((1,), dtype=self.trace.dtype)
    for ind, xvar in enumerate(variables):
        mean[xvar] = np.mean(self.trace[xvar])

    for xind, xvar in enumerate(variables):
        for yind, yvar in enumerate(variables):
            _var = (self.trace[xvar] - mean[xvar]) \
                * (self.trace[yvar] - mean[yvar])
            cov[xind, yind] = np.sum(_var)/float(len(self.trace)-1)
    return cov

def covariance(self):

    _max_str = int(np.max([len(var) for var in self.variables]))

    _dtype = self.trace.dtype.names
    _dtype = [('variable', 'U{}'.format(_max_str))] + \
        [(name, 'float64') for name in _dtype]
    cov = np.empty((len(self.variables),), dtype=_dtype)

    mean = np.empty((1,), dtype=self.trace.dtype)
    for ind, xvar in enumerate(self.variables):
        mean[xvar] = np.mean(self.trace[xvar])

    for ind, xvar in enumerate(self.variables):
        for yvar in self.variables:
            cov[ind]['variable'] = xvar
            _var = (self.trace[xvar] - mean[xvar]) \
                * (self.trace[yvar] - mean[yvar])
            cov[ind][yvar] = np.sum(_var)/float(len(self.trace)-1)
    return cov

