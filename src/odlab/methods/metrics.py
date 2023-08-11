import numpy as np


def diff(sim, obs):
    """Calculates the distance between variable points

    Override this method to include non-standard distance measures
    and coordinate transforms into posterior evaluation
    """
    distances = {obs[var] - sim[var] for var in sim}
    return distances


def azel_diff(sim, obs):
    """Calculates the distances between angles, includes wrapping"""
    daz = obs["az"] - sim["az"]

    daz_tmp = np.mod(obs["az"] + 540.0, 360.0) - np.mod(sim["az"] + 540.0, 360.0)
    inds_ = np.abs(daz) > np.abs(daz_tmp)
    daz[inds_] = daz_tmp[inds_]

    distances = {
        "el": obs["el"] - sim["el"],
        "az": daz,
    }
    return distances
