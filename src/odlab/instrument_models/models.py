#!/usr/bin/env python

"""
Collection of forward models for different instrument types, used for simulating
and performing custom orbit determination for these instruments

"""
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


def get_model(data, model_name, **kwargs):
    Model = MODELS[model_name]
    return Model(data, **kwargs)


def source_to_model(df, model_name, **kwargs):
    """Generate model from data source, throw if not enough data"""
    Model = MODELS[model_name]
    data = {key: df.attrs[key] for key in Model.INPUT_DATA}
    return Model(data, **kwargs)


class ForwardModel:
    OUTPUT_DATA = []
    INPUT_DATA = []

    def __init__(self, data, **kwargs):
        for key in self.INPUT_DATA:
            if key not in data:
                raise ValueError(
                    "Data field {} is mandatory for {}".format(key, type(self).__name__)
                )

        self.data = data

    def evaluate(self, state, **kwargs):
        """Evaluate forward model"""
        raise NotImplementedError()
