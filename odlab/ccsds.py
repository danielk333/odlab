#!/usr/bin/env python

'''CCSDS file readers

https://public.ccsds.org/Pubs/503x0b1c1.pdf
'''

#Python standard import

#Third party import
import numpy as np
import scipy.constants as consts

#Local import



def read_oem(fname):
    meta = {'COMMENT': ''}

    _dtype = [
        ('date', 'datetime64[us]'),
        ('x', 'float64'),
        ('y', 'float64'),
        ('z', 'float64'),
        ('vx', 'float64'),
        ('vy', 'float64'),
        ('vz', 'float64'),
    ]

    raw_data = []
    DATA_ON = False
    META_ON = False
    with open(fname, 'r') as f:
        for line in f:

            if META_ON:
                if line.strip() == 'META_STOP':
                    META_ON = False
                    DATA_ON = True
                    continue

                _tmp = [x.strip() for x in line.split('=')]
                meta[_tmp[0]] = _tmp[1]
            elif DATA_ON:
                if line[:7] == 'COMMENT':
                    meta['COMMENT'] += line[7:]
                else:
                    raw_data.append(line.split(' '))
            else:
                if line.strip() == 'META_START':
                    META_ON = True
                    continue
                _tmp = [x.strip() for x in line.split('=')]
                meta[_tmp[0]] = _tmp[1]


    data_len = len(raw_data)

    data = np.empty((data_len, ), dtype=_dtype)

    for ind, row in enumerate(raw_data):
        rown = 0
        for col, dtype in _dtype:
            data[ind][col] = row[rown]
            rown += 1

    return data, meta



def read_ccsds(fname):
    '''Just get the range data # TODO: the rest
    '''
    meta = {'COMMENT': ''}

    RANGE_UNITS = 'km'
    with open(fname, 'r') as f:
        DATA_ON = False
        META_ON = False
        data_raw = {}
        for line in f:
            if line.strip() == 'DATA_STOP':
                break

            if META_ON:
                tmp_lin = line.split('=')
                if len(tmp_lin) > 1:
                    meta[tmp_lin[0].strip()] = tmp_lin[1].strip()
                    if tmp_lin[0].strip() == 'RANGE_UNITS':
                        RANGE_UNITS = tmp_lin[1].strip().lower()
            elif DATA_ON:
                name, tmp_dat = line.split('=')

                name = name.strip().lower()
                tmp_dat = tmp_dat.strip().split(' ')

                if name in data_raw:
                    data_raw[name].append(tmp_dat)
                else:
                    data_raw[name] = [tmp_dat]
            else:
                if line.lstrip()[:7] == 'COMMENT':
                    meta['COMMENT'] += line.lstrip()[7:]
                else:
                    tmp_lin = line.split('=')
                    if len(tmp_lin) > 1:
                        meta[tmp_lin[0].strip()] = tmp_lin[1].strip()

            if line.strip() == 'META_START':
                META_ON = True
            if line.strip() == 'DATA_START':
                META_ON = False
                DATA_ON = True
    _dtype = [
        ('date', 'datetime64[us]'),
    ]

    data_len = len(data_raw[list(data_raw.keys())[0]])

    for name in data_raw:
        _dtype.append( (name, 'float64') )
        _dtype.append( (name + '_err', 'float64') )

    data = np.empty((data_len, ), dtype=_dtype)

    date_set = False
    for name, series in data_raw.items():
        for ind, val in enumerate(series):
            if not date_set:
                data[ind]['date'] = np.datetime64(val[0],'us')

            data[ind][name] = np.float64(val[1])
            if len(val) > 2:
                data[ind][name + '_err'] = np.float64(val[2])
            else:
                data[ind][name + '_err'] = 0.0

            if name == 'range':
                if RANGE_UNITS == 's':
                    data[ind][name] *= consts.c*1e-3

        date_set = True

    return data, meta


