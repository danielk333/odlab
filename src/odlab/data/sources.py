"""

Defines the supported data sources, each loader must produce a pandas DataFrame

convention is:
there is always a date column
every variable column as a counterpart with the name "_sd" for standard deviation
of that parameter

"""
import logging
from collections import OrderedDict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SOURCES = OrderedDict()


def build_source(date, meta, **data):
    """Takes in the raw components of a source and construct the DataFrame"""
    assert isinstance(date, np.ndarray), "Input date must be numpy array"
    assert np.issubdtype(
        date.dtype, np.datetime64
    ), f"Input date must be in datetime64, not {date.dtype}"
    for key in data:
        assert isinstance(data[key], np.ndarray), "Input data must be numpy array"
        if not key.endswith("_sd"):
            assert (
                key + "_sd" in data
            ), "Every data variable needs an uncertainty variable"

    data["date"] = date
    df = pd.DataFrame(data)
    df.attrs.update(meta)

    return df


def load_source(path, source_type, **kwargs):
    assert source_type in SOURCES, f"{source_type} not found among supported loaders"
    func = SOURCES[source_type]
    return func(path, **kwargs)


def source_loader(name):
    """Decorator to register function as a source loader"""

    def source_wrapper(func):
        logger.debug(f"Registering loader for {name} source")
        assert name not in SOURCES, f"{name} already a registered source loader"
        SOURCES[name] = func
        return func

    return source_wrapper


def glob_sources(path, rules, **kwargs):
    """Load many source files trough glob rules"""
    dfs = []
    for source_type, regex in rules.items():
        files = path.glob(regex)
        dfs += [load_source(file, source_type, **kwargs) for file in files]
    return dfs
