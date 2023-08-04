#!/usr/bin/env python

"""

"""

import numpy as np
import matplotlib.pyplot as plt

from .posterior import _named_to_enumerated


def autocorrelation(results, **kwargs):
    axes = kwargs.get("axes", None)
    if axes is None:
        new_plot = True
        axes = []
    else:
        new_plot = False

    min_k = kwargs.get("min_k", 0)
    max_k = kwargs.get("max_k", len(results.trace) // 100)

    MC_gamma = results.autocovariance(min_k=min_k, max_k=max_k)
    Kv = np.arange(min_k, max_k)

    figs = []
    fig_plots = 6

    for ind, var in enumerate(results.variables):
        if ind % fig_plots == 0:
            if new_plot:
                fig, ax_mat = plt.subplots(
                    3, 2, figsize=(15, 15), sharey=True, sharex=True
                )
                fig.suptitle("Markov Chain autocorrelation functions")
                figs.append(fig)
                ax = [x for xx in ax_mat for x in xx]
                axes.append(ax)

        ax = axes[ind // fig_plots][ind % fig_plots]

        ax.plot(Kv, MC_gamma[var] / MC_gamma[var][0])
        ax.set(
            xlabel="$k$",
            ylabel="$\\hat{\\gamma}_k/\\hat{\\gamma}_0$",
            title='Autocorrelation for "{}"'.format(var),
        )

    return figs, axes


def scatter_trace(results, **kwargs):
    thin = kwargs.get("thin", None)

    km_vars = ["x", "y", "z", "vx", "vy", "vz"]

    trace2 = results.trace.copy()
    for var in km_vars:
        if var in results.variables:
            trace2[var] *= 1e-3
    if thin is not None:
        trace2 = trace2[thin]

    colnames = trace2.dtype.names
    cols_ = len(colnames)

    cols = kwargs.get(
        "columns",
        {
            "x": "x [km]",
            "y": "y [km]",
            "z": "z [km]",
            "vx": "$v_x$ [km/s]",
            "vy": "$v_y$ [km/s]",
            "vz": "$v_z$ [km/s]",
            "A": "A [m$^2$]",
        },
    )
    for key in colnames:
        if key not in cols:
            cols[key] = key

    alpha = kwargs.get("alpha", 0.01)
    size = kwargs.get("size", 1.0)
    figsize = kwargs.get("figsize", (15, 15))
    axes = kwargs.get("axes", None)

    if axes is None:
        fig, axes = plt.subplots(cols_, cols_, figsize=figsize)
    else:
        fig = None

    ax_ranges = {}

    for colx in range(cols_):
        for coly in range(cols_):
            if colx == coly:
                axes[colx][coly].hist(trace2[colnames[colx]])
                ax_ranges[colnames[colx]] = (
                    np.min(trace2[colnames[colx]]),
                    np.max(trace2[colnames[colx]]),
                )
            else:
                axes[colx][coly].scatter(
                    trace2[colnames[coly]],
                    trace2[colnames[colx]],
                    size,
                    alpha=alpha,
                    linewidths=0,
                )

            if coly > 0 and colx + 1 < cols_:
                axes[colx][coly].xaxis.set_visible(False)
                axes[colx][coly].yaxis.set_visible(False)
            elif coly == 0 and colx + 1 < cols_:
                axes[colx][coly].xaxis.set_visible(False)
                axes[colx][coly].set_ylabel(cols[colnames[colx]])
            elif colx + 1 == cols_ and coly > 0:
                axes[colx][coly].yaxis.set_visible(False)
                axes[colx][coly].set_xlabel(cols[colnames[coly]])
            else:
                axes[colx][coly].set_xlabel(cols[colnames[coly]])
                axes[colx][coly].set_ylabel(cols[colnames[colx]])

    reference = kwargs.get("reference", None)

    if reference is not None:
        reference = reference.copy()
        for var in reference.dtype.names:
            if var in km_vars:
                reference[var] *= 1e-3
        for colx in range(cols_):
            for coly in range(cols_):
                if colx == coly:
                    axes[colx][coly].axvline(
                        x=reference[colnames[colx]][0], ymin=0, ymax=1, color="r"
                    )
                else:
                    axes[colx][coly].plot(
                        reference[colnames[coly]][0],
                        reference[colnames[colx]][0],
                        "or",
                    )

    for colx in range(cols_):
        for coly in range(cols_):
            if colx == coly:
                continue
            axes[colx][coly].set_xlim(ax_ranges[colnames[coly]])
            axes[colx][coly].set_ylim(ax_ranges[colnames[colx]])

    if cols_ > 1:
        axes[0][0].set_yticklabels(axes[-1][0].get_xticklabels())

    if fig is not None:
        fig.tight_layout()
        fig.subplots_adjust(hspace=0, wspace=0)

    return fig, axes


def trace(results, **kwargs):
    axes = kwargs.get("axes", None)
    if axes is None:
        new_plot = True
        axes = []
    else:
        new_plot = False

    axis_var = kwargs.get("labels", None)
    if axis_var is None:
        axis_var = []
        for var in results.variables:
            if var in ["x", "y", "z", "vx", "vy", "vz"]:
                axis_var += ["${}$ [km]".format(var)]
            else:
                axis_var += [var]

    reference = kwargs.get("reference", None)

    figs = []
    fig_plots = 6

    for ind, var in enumerate(results.variables):
        if var in ["x", "y", "z", "vx", "vy", "vz"]:
            coef = 1e-3
        else:
            coef = 1.0

        if ind % fig_plots == 0:
            if new_plot:
                fig = plt.figure(figsize=(15, 15))
                fig.suptitle(kwargs.get("title", "MCMC trace plot"))
                figs.append(fig)

        if new_plot:
            ax = fig.add_subplot(231 + (ind % fig_plots))
        else:
            ax = axes[ind // fig_plots][ind % fig_plots]

        ax.plot(results.trace[var] * coef)

        if reference is not None:
            ax.axhline(reference[var][0] * coef, 0, 1, color="r")

        ax.set(
            xlabel="Iteration",
            ylabel="{}".format(axis_var[ind]),
        )

    return figs, axes


def earth_grid(ax, num_lat=25, num_lon=50, alpha=0.1, res=100, color="black"):
    lons = np.linspace(-180, 180, num_lon + 1) * np.pi / 180
    lons = lons[:-1]
    lats = np.linspace(-90, 90, num_lat) * np.pi / 180

    lonsl = np.linspace(-180, 180, res) * np.pi / 180
    latsl = np.linspace(-90, 90, res) * np.pi / 180

    r_e = 6371e3
    for lat in lats:
        x = r_e * np.cos(lonsl) * np.cos(lat)
        y = r_e * np.sin(lonsl) * np.cos(lat)
        z = r_e * np.ones(np.size(lonsl)) * np.sin(lat)
        ax.plot(x, y, z, alpha=alpha, linestyle="-", marker="", color=color)

    for lon in lons:
        x = r_e * np.cos(lon) * np.cos(latsl)
        y = r_e * np.sin(lon) * np.cos(latsl)
        z = r_e * np.sin(latsl)
        ax.plot(x, y, z, alpha=alpha, color=color)

    # Hide grid lines
    ax.grid(False)

    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.axis("off")


def orbits(posterior, **kwargs):
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection="3d")
    earth_grid(ax)
    ax.set_title(kwargs.get("title", "Orbit determination: orbital shift"))

    start = kwargs.get("start", None)
    prior = posterior.kwargs["start"]
    end = posterior.results.MAP
    true = kwargs.get("true", None)

    alpha = kwargs.get("alpha", 0.5)
    max_range = kwargs.get("max_range", 10e6)

    states = [start, prior, end, true]

    _label = [
        "Start state: observations",
        "Start state",
        "Prior: observations",
        "Prior",
        "Maximum a Posteriori: observations",
        "Maximum a Posteriori",
        "True: observations",
        "True",
    ]
    _col = [
        "k",
        "b",
        "g",
        "r",
    ]
    for model in posterior._models:
        for ind, state in enumerate(states):
            if state is None:
                continue
            state = _named_to_enumerated(state, posterior.variables)

            states_obs = model.get_states(state)
            _t = model.data["t"]
            model.data["t"] = np.linspace(0, np.max(_t), num=kwargs.get("num", 1000))
            states_ = model.get_states(state)
            model.data["t"] = _t
            ax.plot(
                states_obs[0, :],
                states_obs[1, :],
                states_obs[2, :],
                "." + _col[ind],
                label=_label[ind * 2],
                alpha=alpha,
            )
            ax.plot(
                states_[0, :],
                states_[1, :],
                states_[2, :],
                "-" + _col[ind],
                label=_label[ind * 2 + 1],
                alpha=alpha,
            )
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    ax.legend()

    plots = []
    plots.append(
        {
            "fig": fig,
            "axes": [ax],
        }
    )

    return plots


def residuals(posterior, states, labels, styles, absolute=False, **kwargs):
    axes = kwargs.get("axes", None)

    if axes is None:
        new_plot = True
        axes = []
    else:
        new_plot = False
    figs = []

    residual_data = []
    for state in states:
        residual_data.append(posterior.residuals(state))

    plot_n = len(posterior._models)

    num = len(residual_data)

    if plot_n > 3:
        _pltn = 3
    else:
        _pltn = plot_n

    _ind = 0
    units = kwargs.get("units", {})

    for ind in range(plot_n):
        variables = [x[0] for x in posterior._models[ind].dtype]

        if _ind == _pltn or _ind == 0:
            _ind = 0
            if new_plot:
                axes.append([])
                fig = plt.figure(figsize=(15, 15))
                fig.suptitle(kwargs.get("title", "Orbit determination residuals"))
                figs.append(fig)

        for vari, var in enumerate(variables):
            if new_plot:
                subpn = 100 * _pltn + len(variables) * 10
                subpn += 1 + vari + _ind * len(variables)
                ax = fig.add_subplot(subpn)
                axes[-1].append(ax)
            else:
                ax = axes[ind][_ind]

            for sti in range(num):
                d0 = residual_data[sti][ind]["date"][0]
                df = residual_data[sti][ind]["date"] - d0
                if absolute:
                    ax.semilogy(
                        df / np.timedelta64(1, "h"),
                        np.abs(residual_data[sti][ind]["residuals"][var]),
                        styles[sti],
                        label=labels[sti],
                        alpha=kwargs.get("alpha", 0.5),
                    )
                else:
                    ax.plot(
                        df / np.timedelta64(1, "h"),
                        residual_data[sti][ind]["residuals"][var],
                        styles[sti],
                        label=labels[sti],
                        alpha=kwargs.get("alpha", 0.5),
                    )
            ax.set(
                xlabel="Time [h]",
                ylabel=f'{var} residuals [{units.get(var, "")}]',
                title="Model {}".format(ind),
            )
            ax.legend()

        _ind += 1

    return figs, axes
