import numpy as np

from .models import ForwardModel, register_model


@register_model("estimated_state")
class EstimatedState(ForwardModel):
    dtype = [
        ("x", "float64"),
        ("y", "float64"),
        ("z", "float64"),
        ("vx", "float64"),
        ("vy", "float64"),
        ("vz", "float64"),
    ]

    def __init__(self, data, propagator, **kwargs):
        super(EstimatedState, self).__init__(data, propagator, **kwargs)

    def evaluate(self, state, **kw):
        """Evaluate forward model"""

        states = self.get_states(state, **kw)

        sim_dat = np.empty((len(self.data["t"]),), dtype=EstimatedState.dtype)

        for ind in range(len(self.data["t"])):
            for dim, npd in enumerate(EstimatedState.dtype):
                name, _ = npd
                sim_dat[ind][name] = states[dim, ind]

        return sim_dat
