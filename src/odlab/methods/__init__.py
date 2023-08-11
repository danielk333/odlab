"""
The models in the native implementation assume a input list of data tuples
structured as
```python
    measurements = [
        (ForwardModel, [DataFrames]),
        (ForwardModel, [DataFrames]),
        (ForwardModel, [DataFrames]),
    ]
```
"""

from . import posterior
from . import solvers

from .solvers import SOLVERS
from .posterior import POSTERIORS
from .state_generator import StateGenerator, sortsPropagator