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
        self.data["geo"] = sorts.frames.ITRS_to_geodetic(
            self.data["st_ecef"][0],
            self.data["st_ecef"][1],
            self.data["st_ecef"][2],
            degrees=True,
        )

    def evaluate(self, t, states, **kwargs):
        """Evaluate forward model"""
        sim_dat = {}

        x = states[:3, :] - self.data["st_ecef"][:3, None]
        r = sorts.frames.ecef_to_enu(
            self.data["geo"][0],
            self.data["geo"][1],
            self.data["geo"][2],
            x,
            degrees=True,
        )
        azel = sorts.frames.cart_to_sph(r, degrees=True)

        sim_dat["az"] = azel[0, :]
        sim_dat["el"] = azel[1, :]

        return sim_dat
