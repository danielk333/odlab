from pathlib import Path
import numpy as np
import pandas as pd
import h5py

from .sources import source_loader
from .. import times


@source_loader("radar_hdf")
def load_radar_hdfs(path, **kwargs):
    path = Path(path)

    with h5py.File(path, 'r') as hf:

        ometa = {}
        sort_obs = np.argsort(hf["m_time"][()])

        data = {}

        data['date'] = times.unix2npdt(
            hf["m_time"][()][sort_obs]
        )
        data['r'] = hf["m_range"][()]*1e3
        data['v'] = hf["m_range_rate"][()]*1e3

        data['r_sd'] = hf["m_range_std"][()]*1e3
        data['v_sd'] = hf["m_range_rate_std"][()]*1e3

        ometa['path'] = path
        ometa['fname'] = path.name
        ometa['tx_ecef'] = hf["tx_loc"][()]
        ometa['rx_ecef'] = hf["rx_loc"][()]

    df = pd.DataFrame(data)
    df.attrs.update(ometa)

    return df
