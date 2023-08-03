from pathlib import Path
import numpy as np
import pandas as pd

import sorts

from .sources import source_loader


@source_loader("text_oem")
def load_text_oem(path, state_cov=None, **kwargs):
    path = Path(path)
    odata, ometa = sorts.io.read_txt_oem(path)
    sort_obs = np.argsort(odata['date'])
    odata = odata[sort_obs]

    data = {}

    for name in odata.dtype.names:
        data[name] = odata[name]

    if state_cov is not None:
        data['cov'] = state_cov
    else:
        data['cov'] = np.diag([1e3]*3 + [10.0]*3)

    # no cov is given for OEM so just assume constant or use user input
    for ind, var in enumerate(['x', 'y', 'z', 'vx', 'vy', 'vz']):
        for rowi in range(len(data)):
            data[var + '_sd'][rowi] = data['cov'][rowi][ind, ind]

    ometa['path'] = path
    ometa['fname'] = path.name
    ometa['frame'] = ometa['REF_FRAME']

    df = pd.DataFrame(data)
    df.attrs.update(ometa)

    return df
