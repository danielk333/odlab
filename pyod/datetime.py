#!/usr/bin/env python

'''

'''

import time
import os

import numpy as np


def npdt2date(dt):
    '''Converts a numpy datetime64 value to a date tuple

    :param numpy.datetime64 dt: Date and time (UTC) in numpy datetime64 format

    :return: tuple (year, month, day, hours, minutes, seconds, microsecond)
             all except usec are integer
    '''

    t0 = np.datetime64('1970-01-01', 's')
    ts = (dt - t0)/sec

    it = int(np.floor(ts))
    usec = 1e6 * (dt - (t0 + it*sec))/sec

    tm = time.localtime(it)
    return tm.tm_year, tm.tm_mon, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec, usec


def npdt2mjd(dt):
    '''Converts a numpy datetime64 value (UTC) to a modified Julian date
    '''
    return (dt - np.datetime64('1858-11-17'))/np.timedelta64(1, 'D')


def mjd2npdt(mjd):
    '''Converts a modified Julian date to a numpy datetime64 value (UTC)
    '''
    day = np.timedelta64(24*3600*1000*1000, 'us')
    return np.datetime64('1858-11-17') + day * mjd


def unix2npdt(unix):
    '''Converts unix seconds to a numpy datetime64 value (UTC)
    '''
    return np.datetime64('1970-01-01') + np.timedelta64(1000*1000,'us')*unix


def npdt2unix(dt):
    '''Converts a numpy datetime64 value (UTC) to unix seconds
    '''
    return (dt - np.datetime64('1970-01-01'))/np.timedelta64(1,'s')

