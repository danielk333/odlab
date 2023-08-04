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

from .maximize_posterior import MaximizeGaussianErrorPosterior
from .state_generator import StateGenerator, sortsPropagator