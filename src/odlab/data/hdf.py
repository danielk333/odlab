from pathlib import Path
import numpy as np
import pandas as pd
import h5py
from astropy.time import Time

from .sources import source_loader


@source_loader("radar_hdf")
def load_radar_hdfs(path, **kwargs):
    path = Path(path)

    with h5py.File(path, "r") as hf:
        ometa = {}
        sort_obs = np.argsort(hf["m_time"][()])

        data = {}

        data["date"] = Time(hf["m_time"][()][sort_obs], format="unix", scale="utc").datetime64
        data["r"] = hf["m_range"][()] * 1e3
        data["v"] = hf["m_range_rate"][()] * 1e3

        data["r_sd"] = hf["m_range_std"][()] * 1e3
        data["v_sd"] = hf["m_range_rate_std"][()] * 1e3

        ometa["path"] = path
        ometa["fname"] = path.name
        ometa["tx_ecef"] = hf["tx_loc"][()]
        ometa["rx_ecef"] = hf["rx_loc"][()]

    df = pd.DataFrame(data)
    df.attrs.update(ometa)

    return df


def save_radar_hdfs(path, df):
    path = Path(path)

    with h5py.File(path, "w") as hf:
        date = Time(df["date"].values, format="datetime64", scale="utc")
        hf["m_time"] = date.unix

        hf["m_range"] = df["r"].values * 1e-3
        hf["m_range_rate"] = df["v"].values * 1e-3

        hf["m_range_std"] = df["r_sd"].values * 1e-3
        hf["m_range_rate_std"] = df["v_sd"].values * 1e-3

        hf["tx_loc"] = df.attrs["tx_ecef"]
        hf["rx_loc"] = df.attrs["rx_ecef"]
