import logging

from tqdm import tqdm
import scipy.optimize as optimize
import numpy as np

from .solvers import Solver, register_solver

logger = logging.getLogger(__name__)

try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
except ImportError:

    class COMM_WORLD:
        size = 1
        rank = 0

    comm = COMM_WORLD()


@register_solver("scipy_maximize")
class ScipyMaximize(Solver):
    OPTIONS = {
        "method": "Nelder-Mead",
        "scipy_options": {},
        "bounds": None,
        "maxiter": 3000,
        "ignore_warnings": False,
    }

    def run(self, posterior, start):
        maxiter = self.options["maxiter"]

        def fun(x):
            val = posterior.logposterior(x)

            pbar.update(1)
            pbar.set_description("Posterior value = {:<10.3f} ".format(val))

            return -val

        logger.info(
            "\n{} running {}".format(type(self).__name__, self.options["method"])
        )

        if self.options["ignore_warnings"]:
            np.seterr(all="ignore")

        pbar = tqdm(total=maxiter, ncols=100, position=comm.rank)
        xhat = optimize.minimize(
            fun,
            start,
            method=self.options["method"],
            options=self.options["scipy_options"],
            bounds=self.options["bounds"],
        )
        pbar.close()

        if self.options["ignore_warnings"]:
            np.seterr(all=None)

        return xhat
