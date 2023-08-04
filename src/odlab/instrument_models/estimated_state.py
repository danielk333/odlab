from .models import ForwardModel, register_model


@register_model("estimated_state")
class EstimatedState(ForwardModel):
    OUTPUT_DATA = [
        "x", "y", "z",
        "vx", "vy", "vz",
    ]

    INPUT_DATA = []

    def evaluate(self, t, states, **kwargs):
        """Evaluate forward model"""
        sim_dat = {
            key: states[ind, :]
            for ind, key in enumerate(self.OUTPUT_DATA)
        }
        return sim_dat
