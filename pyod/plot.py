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

#Local import
from .posterior import _enumerated_to_named
from .posterior import _named_to_enumerated

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

    return fig, ax




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

    _ind = 0
    for ind in range(plot_n):
        if _ind == _pltn or _ind == 0:
            _ind = 0
            fig = plt.figure(figsize=(15,15))
            fig.suptitle(kwargs.get('title', 'Orbit determination residuals'))

        ax = fig.add_subplot(100*_pltn + 21 + _ind*2)
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
