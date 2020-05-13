#!/usr/bin/env python

'''

'''

#Python standard import


#Third party import
import numpy as np

#Local import



# Constants defined by the World Geodetic System 1984 (WGS84)
WGS84_a = 6378.137*1e3
WGS84_esq = 6.69437999014 * 0.001

def geodetic2ecef(lat, lon, alt):
    """
    Convert geodetic coordinates to ECEF.
    @lat, @lon in decimal degrees
    @alt in meters

    Uses WGS84.
    """
    lat, lon = np.radians(lat), np.radians(lon)
    xi = np.sqrt(1 - WGS84_esq * np.sin(lat)**2)
    x = (WGS84_a / xi + alt) * np.cos(lat) * np.cos(lon)
    y = (WGS84_a / xi + alt) * np.cos(lat) * np.sin(lon)
    z = (WGS84_a / xi * (1 - WGS84_esq) + alt) * np.sin(lat)
    return np.array([x, y, z])