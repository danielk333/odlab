#!/usr/bin/env python

'''

'''

#Python standard import


#Third party import
import numpy as np

#Local import



# Constants defined by the World Geodetic System 1984 (WGS84)
WGS84_a = 6378.137*1e3
WGS84_b = 6356.7523142*1e3
WGS84_esq = 6.69437999014 * 0.001
WGS84_e1sq = 6.73949674228 * 0.001


def geodetic2ecef(lat, lon, alt, radians=False):
    """Convert WGS84 geodetic coordinates to ECEF coordinates.
    
    :param float lat: Geographic latitude [deg]
    :param float lon: Geographic longitude [deg]
    :param float alt: Geographic altitude [deg]
    
    :rtype: np.ndarray
    :return: [x, y, z] in ECEF coordinates

    **References:**

        * J. Zhu, "Conversion of Earth-centered Earth-fixed coordinates to geodetic coordinates," IEEE Transactions on Aerospace and Electronic Systems, vol. 30, pp. 957-961, 1994.
    
    """
    if not radians:
        lat, lon = np.radians(lat), np.radians(lon)
    xi = np.sqrt(1 - WGS84_esq * np.sin(lat)**2)
    x = (WGS84_a / xi + alt) * np.cos(lat) * np.cos(lon)
    y = (WGS84_a / xi + alt) * np.cos(lat) * np.sin(lon)
    z = (WGS84_a / xi * (1 - WGS84_esq) + alt) * np.sin(lat)
    return np.array([x, y, z])


def ecef2geodetic(x, y, z, radians=False):
    """Convert ECEF coordinates to WGS84 geodetic coordinates.
        
    :param float x: Position along prime meridian [m]
    :param float y: Position along prime meridian + 90 degrees [m]
    :param float z: Position along earth rotation axis [m]
    
    :rtype: np.ndarray
    :return: [lat [deg], lon [deg], alt [m]] in WGS84 geodetic coordinates

    **References:**

        * J. Zhu, "Conversion of Earth-centered Earth-fixed coordinates to geodetic coordinates," IEEE Transactions on Aerospace and Electronic Systems, vol. 30, pp. 957-961, 1994.
    
    """
    r = np.sqrt(x * x + y * y)
    if r < 1e-9:
        h = np.abs(z) - WGS84_b
        lat = np.sign(z)*np.pi/2
        lon = 0.0
    else:
        Esq = WGS84_a * WGS84_a - WGS84_b * WGS84_b
        F = 54 * WGS84_b * WGS84_b * z * z
        G = r * r + (1 - WGS84_esq) * z * z - WGS84_esq * Esq
        C = (WGS84_esq * WGS84_esq * F * r * r) / (np.power(G, 3))
        S = np.cbrt(1 + C + np.sqrt(C * C + 2 * C))
        P = F / (3 * np.power((S + 1 / S + 1), 2) * G * G)
        Q = np.sqrt(1 + 2 * WGS84_esq * WGS84_esq * P)
        r_0 =  -(P * WGS84_esq * r) / (1 + Q) + np.sqrt(0.5 * WGS84_a * WGS84_a*(1 + 1.0 / Q) - \
            P * (1 - WGS84_esq) * z * z / (Q * (1 + Q)) - 0.5 * P * r * r)
        U = np.sqrt(np.power((r - WGS84_esq * r_0), 2) + z * z)
        V = np.sqrt(np.power((r - WGS84_esq * r_0), 2) + (1 - WGS84_esq) * z * z)
        Z_0 = WGS84_b * WGS84_b * z / (WGS84_a * V)
        h = U * (1 - WGS84_b * WGS84_b / (WGS84_a * V))
        lat = np.arctan((z + WGS84_e1sq * Z_0) / r)
        lon = np.arctan2(y, x)

    if not radians:
        lat, lon = np.degrees(lat), np.degrees(lon)

    return np.array([lat, lon, h])


def ecef2local(lat, lon, alt, x, y, z, radians=False):
    '''
    NEU (east,north,up) from ECEF coordinate system conversion.
    
    TODO: Check the use of alt
    '''
    if not radians:
        lat, lon = np.radians(lat), np.radians(lon)

    mx = np.array([[-np.sin(lon), -np.sin(lat) * np.cos(lon), np.cos(lat) * np.cos(lon)],
                [np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat) * np.sin(lon)],
                [0, np.cos(lat), np.sin(lat)]])
    enu = np.array([x, y, z])
    res = np.dot(np.linalg.inv(mx),enu)
    return res


def cart2azel(vec, radians=False):
    '''Convert from Cartesian coordinates (east,north,up) to spherical in a degrees east of north and elevation fashion

    '''
    x = vec[0]
    y = vec[1]
    z = vec[2]
    r_ = np.sqrt(x**2 + y**2)
    if r_ < 1e-9:
        el = np.sign(z)*numpy.pi*0.5
        az = 0.0
    else:
        el = np.arctan(z/r_)
        az = np.pi/2 - np.arctan2(y,x)
    if radians:
        return np.array([az, el, np.sqrt(x**2 + y**2 +z**2)])
    else:
        return np.array([np.degrees(az), np.degrees(el), np.sqrt(x**2 + y**2 +z**2)])


def azel2cart(vec, radians=False):
    '''Convert from spherical coordinates to Cartesian (east,north,up) in a degrees east of north and elevation fashion

    '''
    _az = vec[0]
    _el = vec[1]
    if not radians:
        _az, _el = np.radians(_az), np.radians(_el)
    return vec[2]*np.array([np.sin(_az)*np.cos(_el), np.cos(_az)*np.cos(_el), np.sin(_el)])
