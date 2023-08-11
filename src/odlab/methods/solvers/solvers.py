import logging
from collections import OrderedDict

logger = logging.getLogger(__name__)

SOLVERS = OrderedDict()


def register_solver(name):
    '''Decorator to register class as a model
    '''

    def solver_wrapper(cls):
        logger.debug(f'Registering solver {name}')
        assert name not in SOLVERS, f"{name} already a registered solver"
        SOLVERS[name] = cls
        return cls

    return solver_wrapper


class Solver:
    OPTIONS = {}

    def __init__(self, **kwargs):
        self.options = {}
        self.options.update(self.OPTIONS)
        self.options.update(kwargs)

    def run(self, posterior, **kwargs):
        raise NotImplementedError("Implement this to construct a method")
