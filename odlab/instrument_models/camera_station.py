import numpy as np

import sorts

from .models import ForwardModel, register_model


@register_model("camera")
class Camera(ForwardModel):
    dtype = [
        ("az", "float64"),
        ("el", "float64"),
    ]

    REQUIRED_DATA = ForwardModel.REQUIRED_DATA + [
        "ecef",
    ]

    def __init__(self, data, propagator, **kwargs):
        super(Camera, self).__init__(data, propagator, **kwargs)

    @staticmethod
    def generate_measurements(state_ecef, ecef, lat, lon):
        x = state_ecef[:3] - ecef
        r = sorts.frames.ecef_to_enu(lat, lon, 0.0, x[0], x[1], x[2])
        azel = sorts.frames.cart_to_sph(r)

        return azel[0], azel[1]

    def distance(self, sim, obs):
        """Calculates the distances between angles, includes wrapping"""
        distances = np.empty(sim.shape, dtype=sim.dtype)

        daz = obs["az"] - sim["az"]
        daz_tmp = np.mod(obs["az"] + 540.0, 360.0) - np.mod(sim["az"] + 540.0, 360.0)
        inds_ = np.abs(daz) > np.abs(daz_tmp)
        daz[inds_] = daz_tmp[inds_]
        distances["el"] = obs["el"] - sim["el"]
        distances["az"] = daz

        return distances

    def evaluate(self, state, **kw):
        """Evaluate forward model"""

        states = self.get_states(state, **kw)

        geo = sorts.frames.ITRS_to_geodetic(
            self.data["ecef"][0], self.data["ecef"][1], self.data["ecef"][2]
        )

        sim_dat = np.empty((len(self.data["t"]),), dtype=Camera.dtype)

        for ind in range(len(self.data["t"])):
            az_obs, el_obs = Camera.generate_measurements(
                states[:, ind], self.data["ecef"], geo[0], geo[1]
            )
            sim_dat[ind]["az"] = az_obs
            sim_dat[ind]["el"] = el_obs

        return sim_dat
