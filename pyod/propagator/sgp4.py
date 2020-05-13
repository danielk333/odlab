#!/usr/bin/env python

'''SGP4 interface with SORTS++.

    Written in 2018 by Juha Vierinen
    based on code from Jan Siminski, ESA.
    Modified by Daniel Kastinen 2018/2019/2020
    fix this later
'''

#Python standard import


#Third party import
import numpy as np
import scipy.constants as consts

import sgp4.earth_gravity
import sgp4.io
import sgp4.propagation
import sgp4.model


#Local import
from .base import Propagator


class SGP4:
    '''
    The SGP4 class acts as a wrapper around the sgp4 module
    uploaded by Brandon Rhodes (http://pypi.python.org/pypi/sgp4/).

        
    It converts orbital elements into the TLE-like 'satellite'-structure which
    is used by the module for the propagation.

    Notes:
        The class can be directly used for propagation. Alternatively,
        a simple propagator function is provided below.
    '''


    # Geophysical constants (WGS 72 values) for notational convinience
    WGS     = sgp4.earth_gravity.wgs72     # Model used within SGP4
    R_EARTH = WGS.radiusearthkm            # Earth's radius [km]
    GM      = WGS.mu                       # Grav.coeff.[km^3/s^2]
    RHO0    = 2.461e-8                     # Density at q0[kg/m^3]
    Q0      = 120.0                        # Reference height [km]
    S0      = 78.0                         # Reference height [km]

    # Time constants

    MJD_0 = 2400000.5

    def __init__(self, mjd_epoch, mean_elements, B):
        '''
        Initialize SGP4 object from mean orbital elements and
        ballistic coefficient

        Creates a sgp4.model.Satellite object for mean element propagation
        First all units are converted to the ones used for TLEs, then
        they are modified to the sgp4 module standard.
        
        Input
        -----
        mjd_epoch     : epoch as Modified Julian Date (MJD)
        mean_elements : [a0,e0,i0,raan0,aop0,M0]
        B             : Ballistic coefficient ( 0.5*C_D*A/m )
        
        Remarks
        -------
        
        This object is not usable for TLE generation, but only for propagation
        as internal variables are modified by sgp4init.
        '''

        a0     = mean_elements[0]         # Semi-major (a') at epoch [km]
        e0     = mean_elements[1]         # Eccentricity at epoch
        i0     = mean_elements[2]         # Inclination at epoch
        raan0  = mean_elements[3]         # RA of the ascending node at epoch
        aop0   = mean_elements[4]         # Argument of perigee at epoch
        M0     = mean_elements[5]         # Mean anomaly at epoch
    
        # Compute ballistic coefficient
        bstar  = 0.5*B*SGP4.RHO0 # B* in [1/m]
        n0 = np.sqrt(SGP4.GM) / (a0**1.5)
        
        # Scaling
        n0    = n0*(86400.0/(2*np.pi))          # Convert to [rev/d]
        bstar = bstar*(SGP4.R_EARTH*1000.0)     # Convert from [1/m] to [1/R_EARTH]
    
        # Compute year and day of year
        d   = mjd_epoch - 16480.0               # Days since 1904 Jan 1.0
        y   = int(int(d) / 365.25)                # Number of years since 1904
        doy = d - int(365.25*y)                 # Day of year
        if (y%4==0):
            doy+=1.0
                    
        # Create Satellite object and fill member variables
        sat = sgp4.model.Satellite()
        #Unique satellite number given in the TLE file.
        sat.satnum = 12345
        #Full four-digit year of this element set's epoch moment.
        sat.epochyr = 1904+y
        #Fractional days into the year of the epoch moment.
        sat.epochdays = doy
        #Julian date of the epoch (computed from epochyr and epochdays).
        sat.jdsatepoch = mjd_epoch + SGP4.MJD_0
        
        #First time derivative of the mean motion (ignored by SGP4).
        #sat.ndot
        #Second time derivative of the mean motion (ignored by SGP4).
        #sat.nddot
        #Ballistic drag coefficient B* in inverse earth radii.
        sat.bstar = bstar
        #Inclination in radians.
        sat.inclo = i0
        #Right ascension of ascending node in radians.
        sat.nodeo = raan0
        #Eccentricity.
        sat.ecco = e0
        #Argument of perigee in radians.
        sat.argpo = aop0
        #Mean anomaly in radians.
        sat.mo = M0
        #Mean motion in radians per minute.
        sat.no = n0 / ( 1440.0 / (2.0 *np.pi) )
        #
        sat.whichconst = SGP4.WGS
        
        sat.a = pow( sat.no*SGP4.WGS.tumin , (-2.0/3.0) )
        sat.alta = sat.a*(1.0 + sat.ecco) - 1.0
        sat.altp = sat.a*(1.0 - sat.ecco) - 1.0
    
        sgp4.propagation.sgp4init(SGP4.WGS, 'i', \
            sat.satnum, sat.jdsatepoch-2433281.5, sat.bstar,\
            sat.ecco, sat.argpo, sat.inclo, sat.mo, sat.no,\
            sat.nodeo, sat)

        # Store satellite object and epoch
        self.sat       = sat
        self.mjd_epoch = mjd_epoch
        
    def state(self, mjd):
        '''
        Inertial position and velocity ([m], [m/s]) at epoch mjd
        

        :param float mjd: epoch where satellite should be propagated to
        
        '''
        # minutes since reference epoch
        m = (mjd - self.mjd_epoch) * 1440.0
        r,v = sgp4.propagation.sgp4(self.sat, m)
        return np.hstack((np.array(r),np.array(v)))
        
    def position(self, mjd):
        '''
        Inertial position at epoch mjd
        
        :param float mjd: epoch where satellite should be propagated to
        '''
        return self.state(mjd)[0:3]
        
    def velocity(self, mjd):
        '''
        Inertial velocity at epoch mjd
                
        :param float mjd: epoch where satellite should be propagated to
        '''
        return self.state(mjd)[3:7]


M_earth = SGP4.GM*1e9/consts.G
'''float: Mass of the Earth using the WGS72 convention.
'''

MU_earth = SGP4.GM*1e9
'''float: Standard gravitational parameter of the Earth using the WGS72 convention.
'''

class PropagatorSGP4(Propagator):
    '''Propagator class implementing the SGP4 propagator.

    **Settings**

    :param bool polar_motion: Determines if polar motion should be used in calculating ITRF frame.
    :param str polar_motion_model: String identifying the polar motion model to be used. Options are '80' or '00'.
    :param str out_frame: String identifying the output frame. Options are 'ITRF' or 'TEME'.
    '''

    def __init__(self):
        super(PropagatorSGP4, self).__init__()

        self.settings.update(
            polar_motion=False, 
            polar_motion_model='80', 
            out_frame='ITRF', 
        )


    def propagate(self, t, state0, mjd0, **kwargs):
        '''
        **Implementation:**

        All state-vector units are in meters.

        Keyword arguments contain only information needed for ballistic coefficient :code:`B` used by SGP4. Either :code:`B` or :code:`C_D`, :code:`A` and :code:`m` must be supplied.
        They also contain a option to give angles in radians or degrees. By default input is assumed to be degrees.

        **Frame:**

        The input frame is ECI (TEME) for orbital elements and Cartesian. The output frame is always ECEF.

        :param float B: Ballistic coefficient
        :param float C_D: Drag coefficient
        :param float A: Cross-sectional Area
        :param float m: Mass
        :param bool radians: If true, all angles are assumed to be in radians.
        '''
        t = self._make_numpy(t)

        if 'B' in kwargs:
            B = kwargs['B']
        else:
            B = 0.5*kwargs['C_D']*kwargs['A']/kwargs['m']

        state_c = state0*1e-3#to km

        mean_elements = tle.TEME_to_TLE(state_c, mjd0=mjd0, kepler=False)

        if np.any(np.isnan(mean_elements)):
            raise Exception('Could not compute SGP4 initial state: {}'.format(mean_elements))

        # Create own SGP4 object
        obj = SGP4(mjd0, mean_elements, B)

        mjdates = mjd0 + t/86400.0
        pos=np.zeros([3,t.size])
        vel=np.zeros([3,t.size])

        for mi,mjd in enumerate(mjdates):
            y = obj.state(mjd)
            pos[:,mi] = y[:3]
            vel[:,mi] = y[3:]

        if self.out_frame == 'TEME':
            states=np.empty((6,t.size), dtype=np.float)
            states[:3,:] = pos*1e3
            states[3:,:] = vel*1e3
            return states

        elif self.out_frame == 'ITRF':
            if self.polar_motion:
                PM_data = tle.get_Polar_Motion(dpt.mjd_to_jd(mjdates))
                xp = PM_data[:,0]
                xp.shape = (1,xp.size)
                yp = PM_data[:,1]
                yp.shape = (1,yp.size)
            else:
                xp = 0.0
                yp = 0.0

            ecefs = teme2ecef(t, pos, vel, mjd0=mjd0, xp=xp, yp=yp ,model=self.polar_motion_model)
            ecefs *= 1e3 #to meter
            return ecefs
        else:
            raise Exception('Output frame {} not found'.format(self.out_frame))



import skyfield.sgp4lib as sgp4lib
from astropy import coordinates as coord, units as u
from astropy.time import Time 

# time- J2000 date
# p,v- vectors, result of SGP4 in TEME frame
date= datetime.datetime(2000, 1, 1, 12, 0) + datetime.timedelta(days=time - 2451545)

# Conversion from TEME to ITRS    
p,v= sgp4lib.TEME_to_ITRF(time,np.asarray(p),np.asarray(v)*86400)
v=v/86400

# Conversion from ITRS to J2000    
now = Time(date)
itrs = coord.ITRS(p[0]*u.km, p[1]*u.km, p[2]*u.km, v[0]*u.km/u.s, v[1]*u.km/u.s, v[2]*u.km/u.s, obstime=now)
gcrs = itrs.transform_to(coord.GCRS(obstime=now))
p,v=gcrs.cartesian.xyz.value,gcrs.velocity.d_xyz.value


if __name__ == "__main__":
    import time
    from propagator_base import plot_orbit_3d


    prop = PropagatorSGP4()

    mjd0 = dpt.jd_to_mjd(2457126.2729)
    C_D = 2.3
    m = 8000
    A = 1.0

    t = np.arange(0,24*3600, dtype=np.float)
    
    t0=time.time()

    ecefs = prop.get_orbit(
        t=t, mjd0=mjd0,
        a=7000e3, e=0.0, inc=90.0,
        raan=10, aop=10, mu0=40.0,
        C_D=C_D, m=m, A=A,
    )

    t1=time.time()

    print('exec time: {} sec'.format(t1-t0))
    
    plot_orbit_3d(ecefs)

    prop = PropagatorSGP4(polar_motion=True, polar_motion_model='00')
    
    t0=time.time()

    ecefs = prop.get_orbit(
        t=t, mjd0=mjd0,
        a=7000e3, e=0.0, inc=90.0,
        raan=10, aop=10, mu0=40.0,
        C_D=C_D, m=m, A=A,
    )

    t1=time.time()

    print('exec time: {} sec'.format(t1-t0))

    plot_orbit_3d(ecefs)
