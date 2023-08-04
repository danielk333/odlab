import numpy as np

from .models import ForwardModel, register_model


@register_model("radar_pair")
class RadarPair(ForwardModel):
    dtype = [
        ("r", "float64"),
        ("v", "float64"),
    ]

    REQUIRED_DATA = ForwardModel.REQUIRED_DATA + [
        "tx_ecef",
        "rx_ecef",
    ]

    def __init__(self, data, propagator, **kwargs):
        super(RadarPair, self).__init__(data, propagator, **kwargs)

    @staticmethod
    def generate_measurements(state_ecef, rx_ecef, tx_ecef):
        r_tx = tx_ecef - state_ecef[:3]
        r_rx = rx_ecef - state_ecef[:3]

        r_tx_n = np.linalg.norm(r_tx)
        r_rx_n = np.linalg.norm(r_rx)

        r_sim = r_tx_n + r_rx_n

        v_tx = -np.dot(r_tx, state_ecef[3:]) / r_tx_n
        v_rx = -np.dot(r_rx, state_ecef[3:]) / r_rx_n

        v_sim = v_tx + v_rx

        return r_sim, v_sim

    def evaluate(self, state, **kw):
        """Evaluate forward model"""

        states = self.get_states(state, **kw)

        sim_dat = np.empty((len(self.data["t"]),), dtype=RadarPair.dtype)

        for ind in range(len(self.data["t"])):
            r_obs, v_obs = RadarPair.generate_measurements(
                states[:, ind], self.data["rx_ecef"], self.data["tx_ecef"]
            )
            sim_dat[ind]["r"] = r_obs
            sim_dat[ind]["v"] = v_obs

        return sim_dat
