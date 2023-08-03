#!/usr/bin/env python

"""
Collection of forward models for different instrument types, used for simulating
and performing custom orbit determination for these instruments

"""

import numpy as np

from . import times

import logging
from collections import OrderedDict

logger = logging.getLogger(__name__)

MODELS = OrderedDict()


def register_model(name):
    '''Decorator to register class as a model
    '''

    def model_wrapper(cls):
        logger.debug(f'Registering model {name}')
        assert name not in MODELS, f"{name} already a registered model"
        MODELS[name] = cls
        return cls

    return model_wrapper


def model_factory(df, model_name):
    """Generate model from data source, throw if not enough data"""

    def generate_model(self, Model, **kwargs):

        avalible_data = [dt[0] for dt in self.dtype]
        model_args = Model.REQUIRED_DATA

        given_data = list(kwargs.keys())
        model_data = {}

        for arg in model_args:
            if arg not in given_data:
                if arg in avalible_data:
                    model_data[arg] = self.data[arg]
                elif arg in self.REQUIRED_META:
                    model_data[arg] = self.meta[arg]
                else:
                    raise ValueError(
                        f'Not enough data to construct model: {arg} missing')
            else:
                model_data[arg] = kwargs[arg]
                del kwargs[arg]

        return Model(data = model_data, **kwargs)


class ForwardModel(object):
    dtype = []  # this is the dtype that is returned by the model

    REQUIRED_DATA = [
        "date",
        "date0",
    ]

    def __init__(self, data, propagator, coord="cart", **kwargs):
        for key in self.REQUIRED_DATA:
            if key not in data:
                raise ValueError(
                    "Data field {} is mandatory for {}".format(key, type(self).__name__)
                )

        self.data = data
        self.propagator = propagator
        self.coord = coord

        self.data["mjd0"] = times.npdt2mjd(self.data["date0"])
        t = (self.data["date"] - self.data["date0"]) / np.timedelta64(1, "s")
        self.data["t"] = t

    def get_states(self, state, **kw):
        states = self.propagator.propagate(
            self.data["t"], state, self.data["mjd0"], **kw
        )

        return states

    def evaluate(self, state, **kw):
        """Evaluate forward model"""
        raise NotImplementedError()

    def distance(self, sim, obs):
        """Calculates the distance between variable points

        Override this method to include non-standard distance measures
        and coordinate transforms into posterior evaluation
        """
        distances = np.empty(sim.shape, dtype=sim.dtype)
        for name in sim.dtype.names:
            distances[name] = obs[name] - sim[name]
        return distances
