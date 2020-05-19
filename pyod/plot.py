#!/usr/bin/env python

'''

'''

#Python standard import


#Third party import
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import matplotlib.path as mpath

try:
    from pandas.plotting import scatter_matrix
    import pandas
except ImportError:
    scatter_matrix = None
    pandas = None

#Local import
from .posterior import _enumerated_to_named
from .posterior import _named_to_enumerated



def autocorrelation(results, **kwargs):

    min_k = kwargs.get('min_k', 0)
    max_k = kwargs.get('max_k', len(results.trace)//100)

    MC_gamma = results.autocovariance(min_k = min_k, max_k = max_k)
    Kv = np.arange(min_k, max_k)
    
    fig1, axes1 = plt.subplots(3, 2,figsize=(15,15), sharey=True, sharex=True)
    fig1.suptitle('Markov Chain autocorrelation functions')
    ind = 0
    for xp in range(3):
        for yp in range(2):
            var = results.variables[ind]
            ax = axes1[xp,yp]
            ax.plot(Kv, MC_gamma[var]/MC_gamma[var][0])
            ax.set(
                xlabel='$k$',
                ylabel='$\hat{\gamma}_k/\hat{\gamma}_0$',
                title='Autocorrelation for "{}"'.format(var),
            )
            ind += 1

    fig2, axes2 = plt.subplots(len(results.variables)-6, 1, figsize=(15,15))
    fig2.suptitle('Markov Chain autocorrelation functions')
    for ind, var in enumerate(results.variables[6:]):
        if len(results.variables)-6 > 1:
            ax = axes2[ind]
        else:
            ax = axes2
        ax.plot(Kv, MC_gamma[var]/MC_gamma[var][0])
        ax.set(
            xlabel='$k$',
            ylabel='$\hat{\gamma}_k/\hat{\gamma}_0$',
            title='Autocorrelation for "{}"'.format(var),
        )

    plots = []
    plots.append({
        'fig': fig1,
        'axes': [axes1],
    })
    plots.append({
        'fig': fig2,
        'axes': [axes2],
    })

    return plots




def scatter_trace(results, **kwargs):

    if scatter_matrix is None:
        raise ImportError('pandas package is required for this plotting function')

    thin = kwargs.get('thin', None)

    km_vars = ['x','y','z','vx','vy','vz']

    trace2 = results.trace.copy()
    for var in km_vars:
        if var in results.variables:
            trace2[var] *= 1e-3
    if thin is not None:
        trace2 = trace2[thin]
    df = pandas.DataFrame.from_records(trace2)
    colnames = df.columns.copy()

    cols = kwargs.get('columns', {
        'x':'x [km]',
        'y':'y [km]',
        'z':'z [km]',
        'vx':'$v_x$ [km/s]',
        'vy':'$v_y$ [km/s]',
        'vz':'$v_z$ [km/s]',
        'A':'A [m$^2$]',
    })
    df = df.rename(columns=cols)

    axes = scatter_matrix(df, alpha=kwargs.get('alpha', 0.01), figsize=(15,15))

    reference = kwargs.get('reference', None)

    cols_ = len(colnames)

    if reference is not None:
        reference = reference.copy()
        for var in reference.dtype.names:
            if var in km_vars:
                reference[var] *= 1e-3
        for colx in range(cols_):
            for coly in range(cols_):
                if colx == coly:
                    axes[colx][coly].axvline(x=reference[colnames[colx]][0], ymin=0, ymax=1, color='r')
                else:
                    axes[colx][coly].plot(reference[colnames[colx]][0], reference[colnames[coly]][0], 'or')
    return axes



def trace(results, **kwargs):

    axis_var = kwargs.get('labels', None)
    if axis_var is None:
        axis_var = []
        for var in results.variables:
            if var in ['x', 'y', 'z', 'vx', 'vy', 'vz']:
                axis_var += ['${}$ [km]'.format(var)]
            else:
                axis_var += [var]

    reference = kwargs.get('reference', None)

    plots = []
    for ind, var in enumerate(results.variables):
        if var in ['x', 'y', 'z', 'vx', 'vy', 'vz']:
            coef = 1e-3
        else:
            coef = 1.0
    
        if ind == 0 or ind == 6:
            fig = plt.figure(figsize=(15,15))
            fig.suptitle(kwargs.get('title','MCMC trace plot'))
            plots.append({
                'fig': fig,
                'axes': [],
            })
            
        if ind <= 5:
            ax = fig.add_subplot(231+ind)
        if ind > 5:
            ax = fig.add_subplot(100*(len(results.variables) - 6) + ind - 5 + 10)
        plots[-1]['axes'].append(ax)
        ax.plot(results.trace[var]*coef)

        if reference is not None:
            ax.axhline(reference[var][0]*coef, 0, 1, color='r')

        ax.set(
            xlabel='Iteration',
            ylabel='{}'.format(axis_var[ind]),
        )

    return plots



def earth_grid(ax,num_lat=25,num_lon=50,alpha=0.1,res = 100, color='black'):
    lons = np.linspace(-180, 180, num_lon+1) * np.pi/180 
    lons = lons[:-1]
    lats = np.linspace(-90, 90, num_lat) * np.pi/180 

    lonsl = np.linspace(-180, 180, res) * np.pi/180 
    latsl = np.linspace(-90, 90, res) * np.pi/180 

    r_e=6371e3    
    for lat in lats:
        x = r_e*np.cos(lonsl)*np.cos(lat)
        y = r_e*np.sin(lonsl)*np.cos(lat)
        z = r_e*np.ones(np.size(lonsl))*np.sin(lat)
        ax.plot(x,y,z,alpha=alpha,linestyle='-', marker='',color=color)

    for lon in lons:
        x = r_e*np.cos(lon)*np.cos(latsl)
        y = r_e*np.sin(lon)*np.cos(latsl)
        z = r_e*np.sin(latsl)
        ax.plot(x,y,z,alpha=alpha,color=color)
    
    # Hide grid lines
    ax.grid(False)

    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.axis('off')



def orbits(posterior, **kwargs):

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    earth_grid(ax)
    ax.set_title(kwargs.get('title', 'Orbit determination: orbital shift'))

    start = kwargs.get('start', None)
    prior = posterior.kwargs['start']
    end = posterior.results.MAP
    true = kwargs.get('true', None)

    alpha = kwargs.get('alpha', 0.5)
    max_range = kwargs.get('max_range', 10e6)

    states = [start, prior, end, true]

    _label = [
        'Start state: observations',
        'Start state',
        'Prior: observations',
        'Prior',
        'Maximum a Posteriori: observations',
        'Maximum a Posteriori',
        'True: observations',
        'True',
    ]
    _col = [
        'k',
        'b',
        'g',
        'r',
    ]
    for model in posterior._models:
        for ind, state in enumerate(states):

            if state is None:
                continue
            state = _named_to_enumerated(state, posterior.variables)

            states_obs = model.get_states(state)
            _t = model.data['t']
            model.data['t'] = np.linspace(0, np.max(_t), num=kwargs.get('num',1000))
            states_ = model.get_states(state)
            model.data['t'] = _t
            ax.plot(states_obs[0,:], states_obs[1,:], states_obs[2,:],"."+_col[ind],
                label=_label[ind*2], alpha=alpha,
            )
            ax.plot(states_[0,:], states_[1,:], states_[2,:],"-"+_col[ind],
                label=_label[ind*2+1], alpha=alpha,
            )
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    ax.legend()

    plots = []
    plots.append({
        'fig': fig,
        'axes': [ax],
    })

    return plots




def residuals(posterior, states, labels, styles, absolute=False, **kwargs):

    residuals = []
    for state in states:
        residuals.append(
            posterior.residuals(state)
        )
    
    plot_n = len(residuals[-1])

    num = len(residuals)

    if plot_n > 3:
        _pltn = 3
    else:
        _pltn = plot_n

    plots = []

    _ind = 0
    for ind in range(plot_n):
        if _ind == _pltn or _ind == 0:
            _ind = 0
            fig = plt.figure(figsize=(15,15))
            fig.suptitle(kwargs.get('title', 'Orbit determination residuals'))
            plots.append({
                'fig': fig,
                'axes': [],
            })

        ax = fig.add_subplot(100*_pltn + 21 + _ind*2)
        plots[-1]['axes'].append(ax)

        for sti in range(num):
            if absolute:
                lns = ax.semilogy(
                    (residuals[sti][ind]['date'] - residuals[sti][ind]['date'][0])/np.timedelta64(1,'h'),
                    np.abs(residuals[sti][ind]['residuals']['r']),
                    styles[sti], label=labels[sti], alpha = kwargs.get('alpha',0.5),
                )
            else:
                lns = ax.plot(
                    (residuals[sti][ind]['date'] - residuals[sti][ind]['date'][0])/np.timedelta64(1,'h'),
                    residuals[sti][ind]['residuals']['r'],
                    styles[sti], label=labels[sti], alpha = kwargs.get('alpha',0.5),
                )
        ax.set(
            xlabel='Time [h]',
            ylabel='Range residuals [m]',
            title='Model {}'.format(ind),
        )
        ax.legend()

        ax = fig.add_subplot(100*_pltn + 21+_ind*2+1)
        plots[-1]['axes'].append(ax)

        for sti in range(num):
            if absolute:
                lns = ax.semilogy(
                    (residuals[sti][ind]['date'] - residuals[sti][ind]['date'][0])/np.timedelta64(1,'h'),
                    np.abs(residuals[sti][ind]['residuals']['v']),
                    styles[sti], label=labels[sti], alpha = kwargs.get('alpha',0.5),
                )
            else:
                lns = ax.plot(
                    (residuals[sti][ind]['date'] - residuals[sti][ind]['date'][0])/np.timedelta64(1,'h'),
                    residuals[sti][ind]['residuals']['v'],
                    styles[sti], label=labels[sti], alpha = kwargs.get('alpha',0.5),
                )
        ax.set(
            xlabel='Time [h]',
            ylabel='Velocity residuals [m/s]',
            title='Model {}'.format(ind),
        )
        _ind += 1

    return plots