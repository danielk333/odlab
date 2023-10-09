import pyant
import sorts

from .models import ForwardModel, register_model


@register_model("camera")
class Camera(ForwardModel):
    OUTPUT_DATA = [
        "az",
        "el",
    ]

    INPUT_DATA = [
        "st_ecef",
    ]

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        self.data["ecef_lla"] = pyant.coordinates.cart_to_sph(
            self.data["st_ecef"],
            degrees=True,
        )
        self.data["ecef_lla"][0] = 90 - self.data["ecef_lla"][0]

    def evaluate(self, t, states, **kwargs):
        """Evaluate forward model"""
        sim_dat = {}

        rel_ = states.copy()
        rel_[:3, :] = rel_[:3, :] - self.data["st_ecef"][:3, None]
        rel_[:3, :] = sorts.frames.ecef_to_enu(
            lat = self.data["ecef_lla"][1],
            lon = self.data["ecef_lla"][0],
            alt = self.data["ecef_lla"][2],
            ecef = rel_[:3, :],
            degrees=True,
        )
        azel = pyant.coordinates.cart_to_sph(rel_[:3, :], degrees=True)
        sim_dat["az"] = azel[0, :]
        sim_dat["el"] = azel[1, :]

        return sim_dat
