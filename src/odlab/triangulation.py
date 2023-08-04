#! /usr/bin/env python3

'''Module for performing triangulation
'''

import numpy as np


def calculate_linear_system(directions, points):
    '''Calculate the linear system for finding the point closest to N lines.

    Parameters
    ----------
    directions : numpy.array
        (3, N) array of normalized line directions
    points : numpy.array
        (3, N) array of points on the lines

    Returns
    -------
    (numpy.array, numpy.array) -
        (M, b) The (3, 3) matrix and (3,) vector in the equation system $M x = b$

    Denote a point on the line $i$ as $\\mathbf{a}_i$,
    the normalized line direction as $\\mathbf{d}_i$,
    and the point to be solved for as $\\mathbf{p}$.
    Then, to solve for the point $\\mathbf{p}$ that is closest to all lines,
    we start from the sum squared distance to all lines from this point

    $$
        D = \\Sum_{i}^N | \\mathbf{d}_i \\cross (\\mathbf{a}_i - \\mathbf{p}) |^2.
    $$

    Solving $\\nabla D = \\mathbf{0}$ yilds an equation system of the form
    $ M \\mathbf{x} = \\mathbf{b} $.

    This function computes $M$ and $\\mathbf{b}$.
    '''
    M = np.zeros((3, 3))
    b = np.zeros((3, ))
    Im = np.eye(3)

    for ind in range(directions.shape[1]):
        d = directions[:, ind]
        a = points[:, ind]
        da = np.dot(d, a)

        M += np.outer(d, d) - Im
        b += da*d - a

    return M, b


def solve_triangulation(directions, points):
    '''Compute the linear system for finding the point closest to N lines
    and solve that system.

    Parameters
    ----------
    directions : numpy.array
        (3, N) array of normalized line directions
    points : numpy.array
        (3, N) array of points on the lines

    Returns
    -------
    (numpy.array) -
        (x) The (3,) vector closest to all input lines.

    '''
    system_mat, system_result = calculate_linear_system(directions, points)
    closest_point = np.linalg.solve(system_mat, system_result)

    return closest_point


if __name__ == "__main__":

    stations = np.array(
        [
            [1, 2, 0, 0],
            [0, 1, -1, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.float64
    )
    target = np.array([3, 0, 20])

    dirs = target[:, None] - stations
    dirs_norm = dirs/np.linalg.norm(dirs, axis=0)
    point = solve_triangulation(dirs_norm, stations)

    print('NO INPUT VARIANCE')
    print('target = ', target)
    print('point = ', point)
    print('point - target = ', point - target)
    print('Error: ', np.linalg.norm(point - target))

    dirs = (target[:, None] + np.random.randn(*stations.shape)*0.05) - stations
    dirs_norm = dirs/np.linalg.norm(dirs, axis=0)
    point = solve_triangulation(dirs_norm, stations)

    print('INCLUDE INPUT VARIANCE')
    print('target = ', target)
    print('point = ', point)
    print('point - target = ', point - target)
    print('Error: ', np.linalg.norm(point - target))
