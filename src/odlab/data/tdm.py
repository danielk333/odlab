from pathlib import Path
import numpy as np
import pandas as pd

import sorts

from .sources import source_loader


@source_loader("radar_text_tdm")
def load_radar_text_tdm(path, **kwargs):
    path = Path(path)

    odata, ometa = sorts.io.read_txt_tdm(path)
    sort_obs = np.argsort(odata['date'])
    odata = odata[sort_obs]

    data = {}
    data['date'] = odata['date']
    data['r'] = odata['range']*1e3
    data['v'] = odata['doppler_instantaneous']*1e3

    data['r_sd'] = odata['range_err']*1e3
    data['v_sd'] = odata['doppler_instantaneous_err']*1e3

    _cm = ometa['COMMENT'].split('\n')
    for com in _cm:
        tx_ind = com.find('TX_ECEF')
        rx_ind = com.find('RX_ECEF')

        if tx_ind != -1:
            tx_ecef = com[com.find('(')+1: com.find(')')].split(',')
            tx_ecef = np.array([float(x)
                                for x in tx_ecef], dtype=np.float64)
        elif rx_ind != -1:
            rx_ecef = com[com.find('(')+1: com.find(')')].split(',')
            rx_ecef = np.array([float(x)
                                for x in rx_ecef], dtype=np.float64)

    ometa['path'] = path
    ometa['fname'] = path.name
    ometa['tx_ecef'] = tx_ecef
    ometa['rx_ecef'] = rx_ecef

    df = pd.DataFrame(data)
    df.attrs.update(ometa)

    return df
