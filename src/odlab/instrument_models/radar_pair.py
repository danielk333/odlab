import numpy as np

from .models import ForwardModel, register_model


@register_model("radar_pair")
class RadarPair(ForwardModel):
    OUTPUT_DATA = [
        "r",
        "v",
    ]

    INPUT_DATA = [
        "tx_ecef",
        "rx_ecef",
    ]

    def evaluate(self, t, states, **kwargs):
        """Evaluate forward model"""
        sim_dat = {}

        # From SC to station for later use in velocity projection
        r_tx = self.data["tx_ecef"][:3, None] - states[:3, :]
        r_rx = self.data["rx_ecef"][:3, None] - states[:3, :]

        r_tx_n = np.linalg.norm(r_tx, axis=0)
        r_rx_n = np.linalg.norm(r_rx, axis=0)

        r_sim = r_tx_n + r_rx_n
        v_tx = -np.sum(r_tx*states[3:, :], axis=0) / r_tx_n
        v_rx = -np.sum(r_rx*states[3:, :], axis=0) / r_rx_n

        v_sim = v_tx + v_rx

        sim_dat["r"] = r_sim
        sim_dat["v"] = v_sim

        return sim_dat
