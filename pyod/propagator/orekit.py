#!/usr/bin/env python

'''Wrapper for the Orekit propagator into SORTS++ format.

**Links:**
    * `orekit <https://www.orekit.org/>`_
    * `orekit python <https://gitlab.orekit.org/orekit-labs/python-wrapper>`_
    * `orekit python guide <https://gitlab.orekit.org/orekit-labs/python-wrapper/wikis/Manual-Installation-of-Python-Wrapper>`_
    * `Hipparchus <https://www.hipparchus.org/>`_
    * `orekit 9.3 api <https://www.orekit.org/static/apidocs/index.html>`_
    * `JCC <https://pypi.org/project/JCC/>`_


**Example usage:**

Simple propagation showing time difference due to loading of model data.

.. code-block:: python

    from propagator_orekit import PropagatorOrekit
 
'''

#Python standard import
import os
import time

#Third party import
import numpy as np
import scipy
import orekit

#Local import
from .base import Propagator
from .. import datetime as datetime_internal


orekit.initVM()

from orekit.pyhelpers import setup_orekit_curdir
from org.orekit.propagation.numerical import NumericalPropagator
from org.orekit.propagation import SpacecraftState
from org.orekit.frames import FramesFactory
from org.orekit.time import TimeScalesFactory, AbsoluteDate
from org.orekit.orbits import KeplerianOrbit, EquinoctialOrbit, CartesianOrbit, PositionAngle
from org.orekit.utils import Constants, IERSConventions, PVCoordinates
from org.orekit.bodies import CelestialBodyFactory, OneAxisEllipsoid
from org.orekit.python import PythonOrekitStepHandler
import org

from orekit import JArray_double

utc = TimeScalesFactory.getUTC()



def npdt2absdate(dt):
    '''
    Converts a numpy datetime64 value to an orekit AbsoluteDate
    '''

    year, month, day, hour, minutes, seconds, microsecond = datetime_internal.npdt2date(dt)
    return AbsoluteDate(
        int(year),
        int(month),
        int(day),
        int(hour),
        int(minutes),
        float(seconds + microsecond*1e-6),
        utc,
    )


def mjd2absdate(mjd):
    '''
    Converts a Modified Julian Date value to an orekit AbsoluteDate
    '''

    return npdt2absdate(datetime_internal.mjd2npdt(mjd))



def _get_frame(name, frame_tidal_effects = False):
    '''Uses a string to identify which coordinate frame to initialize from Orekit package.

    See `Orekit FramesFactory <https://www.orekit.org/static/apidocs/org/orekit/frames/FramesFactory.html>`_
    '''
    if name == 'EME':
        return FramesFactory.getEME2000()
    elif name == 'CIRF':
        return FramesFactory.getCIRF(IERSConventions.IERS_2010, not frame_tidal_effects)
    elif name == 'ITRF':
        return FramesFactory.getITRF(IERSConventions.IERS_2010, not frame_tidal_effects)
    elif name == 'TIRF':
        return FramesFactory.getTIRF(IERSConventions.IERS_2010, not frame_tidal_effects)
    elif name == 'ITRFEquinox':
        return FramesFactory.getITRFEquinox(IERSConventions.IERS_2010, not frame_tidal_effects)
    if name == 'TEME':
        return FramesFactory.getTEME()
    else:
        raise Exception('Frame "{}" not recognized'.format(name))


def iter_states(fun):

    def iter_fun(state, mjd, *args, **kwargs):

        if len(state.shape) > 1:
            if state.shape[1] > 1:
                if len(mjd) > 1:
                    assert len(mjd) == state.shape[1], 'State and MJD lengths do not correspond'
                else:
                    raise Exception('Need a epoch for each state')

                state_out = np.empty(state.shape, dtype=state.dtype)
                for ind in range(state.shape[1]):
                    state_tmp = state[:,ind]
                    state_out[:,ind] = fun(state_tmp, mjd[ind], *args, **kwargs)
                return state_out
        
        return fun(state, mjd, *args, **kwargs)

    return iter_fun


@iter_states
def frame_conversion(state, mjd, orekit_in_frame, orekit_out_frame):
    '''Convert Cartesian state between two frames. Currently options are:

    * EME
    * CIRF
    * ITRF
    * TIRF
    * ITRFEquinox
    * TEME

    # TODO: finish docstring
    '''

    state_out = np.empty(state.shape, dtype=state.dtype)

    pos = org.hipparchus.geometry.euclidean.threed.Vector3D(float(state[0]), float(state[1]), float(state[2]))
    vel = org.hipparchus.geometry.euclidean.threed.Vector3D(float(state[3]), float(state[4]), float(state[5]))
    PV_state = PVCoordinates(pos, vel)

    transform = orekit_in_frame.getTransformTo(orekit_out_frame, mjd2absdate(mjd))
    new_PV = transform.transformPVCoordinates(PV_state)


    x_tmp = new_PV.getPosition()
    v_tmp = new_PV.getVelocity()

    state_out[0] = x_tmp.getX()
    state_out[1] = x_tmp.getY()
    state_out[2] = x_tmp.getZ()
    state_out[3] = v_tmp.getX()
    state_out[4] = v_tmp.getY()
    state_out[5] = v_tmp.getZ()

    return state_out





class PropagatorOrekit(Propagator):
    '''Propagator class implementing the Orekit propagator.


    :ivar list solarsystem_perturbers: List of strings of names of objects in the solarsystem that should be used for third body perturbation calculations. All objects listed at `CelestialBodyFactory <https://www.orekit.org/static/apidocs/org/orekit/bodies/CelestialBodyFactory.html>`_ are available.
    :ivar str in_frame: String identifying the input frame to be used. All frames listed at `FramesFactory <https://www.orekit.org/static/apidocs/org/orekit/frames/FramesFactory.html>`_ are available.
    :ivar str out_frame: String identifying the output frame to be used. All frames listed at `FramesFactory <https://www.orekit.org/static/apidocs/org/orekit/frames/FramesFactory.html>`_ are available.
    :ivar bool drag_force: Should drag force be included in propagation.
    :ivar bool radiation_pressure: Should radiation pressure force be included in propagation.
    :ivar bool frame_tidal_effects: Should coordinate frames include Tidal effects.
    :ivar str integrator: String representing the numerical integrator from the Hipparchus package to use. Any integrator listed at `Hipparchus nonstiff ode <https://www.hipparchus.org/apidocs/org/hipparchus/ode/nonstiff/package-summary.html>`_ is available.
    :ivar float minStep: Minimum time step allowed in the numerical orbit propagation given in seconds.
    :ivar float maxStep: Maximum time step allowed in the numerical orbit propagation given in seconds.
    :ivar float position_tolerance: Position tolerance in numerical orbit propagation errors given in meters.
    :ivar str earth_gravity: Gravitation model to use for calculating central acceleration force. Currently avalible options are `'HolmesFeatherstone'` and `'Newtonian'`. See `gravity <https://www.orekit.org/static/apidocs/org/orekit/forces/gravity/package-summary.html>`_.
    :ivar tuple gravity_order: A tuple of two integers for describing the order of spherical harmonics used in the `HolmesFeatherstoneAttractionModel <https://www.orekit.org/static/apidocs/org/orekit/forces/gravity/HolmesFeatherstoneAttractionModel.html>`_ model.
    :ivar str atmosphere: Atmosphere model used to calculate atmospheric drag. Currently available options are `'DTM2000'`. See `atmospheres <https://www.orekit.org/static/apidocs/org/orekit/forces/drag/atmosphere/package-summary.html>`_.
    :ivar str solar_activity: The model used for calculating solar activity and thereby the influx of solar radiation. Used in the atmospheric drag force model. Currently available options are `'Marshall'` for the `MarshallSolarActivityFutureEstimation <https://www.orekit.org/static/apidocs/org/orekit/forces/drag/atmosphere/data/MarshallSolarActivityFutureEstimation.html>`_.
    :ivar str constants_source: Controls which source for Earth constants to use. Currently avalible options are `'WGS84'` and `'JPL-IAU'`. See `constants <https://www.orekit.org/static/apidocs/org/orekit/utils/Constants.html>`_.
    :ivar float mu: Standard gravitational constant for the Earth.  Definition depend on the :class:`propagator_orekit.PropagatorOrekit` constructor parameter :code:`constants_source`
    :ivar float R_earth: Radius of the Earth in m. Definition depend on the :class:`propagator_orekit.PropagatorOrekit` constructor parameter :code:`constants_source`
    :ivar float f_earth: Flattening of the Earth (i.e. :math:`\\frac{a-b}{a}` ). Definition depend on the :class:`propagator_orekit.PropagatorOrekit` constructor parameter :code:`constants_source`.
    :ivar float M_earth: Mass of the Earth in kg. Definition depend on the :class:`propagator_orekit.PropagatorOrekit` constructor parameter :code:`constants_source`
    :ivar org.orekit.frames.Frame inputFrame: The orekit frame instance for the input frame.
    :ivar org.orekit.frames.Frame outputFrame: The orekit frame instance for the output frame.
    :ivar org.orekit.frames.Frame inertialFrame: The orekit frame instance for the inertial frame. If inputFrame is pseudo innertial this is the same as inputFrame.
    :ivar org.orekit.bodies.OneAxisEllipsoid body: The model ellipsoid representing the Earth.
    :ivar dict _forces: Dictionary of forces to include in the numerical integration. Contains instances of children of :class:`org.orekit.forces.AbstractForceModel`.
    :ivar list _tolerances: Contains the absolute and relative tolerances calculated by the `tolerances <https://www.orekit.org/static/apidocs/org/orekit/propagation/numerical/NumericalPropagator.html#tolerances(double,org.orekit.orbits.Orbit,org.orekit.orbits.OrbitType)>`_ function.
    :ivar org.orekit.propagation.numerical.NumericalPropagator propagator: The numerical propagator instance.
    :ivar org.orekit.forces.drag.atmosphere.data.MarshallSolarActivityFutureEstimation.StrengthLevel SolarStrengthLevel: The strength of the solar activity. Options are 'AVRAGE', 'STRONG', 'WEAK'.

    The constructor creates a propagator instance with supplied options.

    :param list solarsystem_perturbers: List of strings of names of objects in the solarsystem that should be used for third body perturbation calculations. All objects listed at `CelestialBodyFactory <https://www.orekit.org/static/apidocs/org/orekit/bodies/CelestialBodyFactory.html>`_ are available.
    :param str in_frame: String identifying the input frame to be used. All frames listed at `FramesFactory <https://www.orekit.org/static/apidocs/org/orekit/frames/FramesFactory.html>`_ are available.
    :param str out_frame: String identifying the output frame to be used. All frames listed at `FramesFactory <https://www.orekit.org/static/apidocs/org/orekit/frames/FramesFactory.html>`_ are available.
    :param bool drag_force: Should drag force be included in propagation.
    :param bool radiation_pressure: Should radiation pressure force be included in propagation.
    :param bool frame_tidal_effects: Should coordinate frames include Tidal effects.
    :param str integrator: String representing the numerical integrator from the Hipparchus package to use. Any integrator listed at `Hipparchus nonstiff ode <https://www.hipparchus.org/apidocs/org/hipparchus/ode/nonstiff/package-summary.html>`_ is available.
    :param float min_step: Minimum time step allowed in the numerical orbit propagation given in seconds.
    :param float max_step: Maximum time step allowed in the numerical orbit propagation given in seconds.
    :param float position_tolerance: Position tolerance in numerical orbit propagation errors given in meters.
    :param str atmosphere: Atmosphere model used to calculate atmospheric drag. Currently available options are `'DTM2000'`. See `atmospheres <https://www.orekit.org/static/apidocs/org/orekit/forces/drag/atmosphere/package-summary.html>`_.
    :param str solar_activity: The model used for calculating solar activity and thereby the influx of solar radiation. Used in the atmospheric drag force model. Currently available options are `'Marshall'` for the `MarshallSolarActivityFutureEstimation <https://www.orekit.org/static/apidocs/org/orekit/forces/drag/atmosphere/data/MarshallSolarActivityFutureEstimation.html>`_.
    :param str constants_source: Controls which source for Earth constants to use. Currently avalible options are `'WGS84'` and `'JPL-IAU'`. See `constants <https://www.orekit.org/static/apidocs/org/orekit/utils/Constants.html>`_.
    :param str earth_gravity: Gravitation model to use for calculating central acceleration force. Currently avalible options are `'HolmesFeatherstone'` and `'Newtonian'`. See `gravity <https://www.orekit.org/static/apidocs/org/orekit/forces/gravity/package-summary.html>`_.
    :param tuple gravity_order: A tuple of two integers for describing the order of spherical harmonics used in the `HolmesFeatherstoneAttractionModel <https://www.orekit.org/static/apidocs/org/orekit/forces/gravity/HolmesFeatherstoneAttractionModel.html>`_ model.
    :param str solar_activity_strength: The strength of the solar activity. Options are 'AVRAGE', 'STRONG', 'WEAK'.
    '''


    class OrekitVariableStep(PythonOrekitStepHandler):
        '''Class for handling the steps.
        '''
        def set_params(self, t, start_date, states_pointer, outputFrame):
            self.t = t
            self.start_date = start_date
            self.states_pointer = states_pointer
            self.outputFrame = outputFrame

        def init(self, s0, t):
            pass

        def handleStep(self, interpolator, isLast):
            state1 = interpolator.getCurrentState()
            state0 = interpolator.getPreviousState()

            t0 = state0.getDate().durationFrom(self.start_date)
            t1 = state1.getDate().durationFrom(self.start_date)

            for ti, t in enumerate(self.t):

                if np.abs(t) >= np.abs(t0) and np.abs(t) <= np.abs(t1):
                    t_date = self.start_date.shiftedBy(float(t))

                    _state = interpolator.getInterpolatedState(t_date)

                    PVCoord = _state.getPVCoordinates(self.outputFrame)

                    x_tmp = PVCoord.getPosition()
                    v_tmp = PVCoord.getVelocity()

                    self.states_pointer[0,ti] = x_tmp.getX()
                    self.states_pointer[1,ti] = x_tmp.getY()
                    self.states_pointer[2,ti] = x_tmp.getZ()
                    self.states_pointer[3,ti] = v_tmp.getX()
                    self.states_pointer[4,ti] = v_tmp.getY()
                    self.states_pointer[5,ti] = v_tmp.getZ()

    DEFAULT_SETTINGS = dict(
            in_frame='EME',
            out_frame='ITRF',
            frame_tidal_effects=False,
            integrator='DormandPrince853',
            min_step=0.001,
            max_step=120.0,
            position_tolerance=10.0,
            earth_gravity='HolmesFeatherstone',
            gravity_order=(10,10),
            solarsystem_perturbers=['Moon', 'Sun'],
            drag_force=True,
            atmosphere='DTM2000',
            radiation_pressure=True,
            solar_activity='Marshall',
            constants_source='WGS84',
            solar_activity_strength='WEAK',
        )

    def __init__(self,
                orekit_data,
                settings=None,
            ):

        super(PropagatorOrekit, self).__init__()

        self.settings.update(PropagatorOrekit.DEFAULT_SETTINGS)
        if settings is not None:
            self.settings.update(settings)

        setup_orekit_curdir(filename = orekit_data)

        self.__settings = dict()
        self.__settings['SolarStrengthLevel'] = getattr(org.orekit.forces.drag.atmosphere.data.MarshallSolarActivityFutureEstimation.StrengthLevel, solar_activity_strength)
        self._tolerances = None
        
        if self.settings['constants_source'] == 'JPL-IAU':
            self.mu = Constants.JPL_SSD_EARTH_GM
            self.R_earth = Constants.IAU_2015_NOMINAL_EARTH_EQUATORIAL_RADIUS
            self.f_earth = (Constants.IAU_2015_NOMINAL_EARTH_EQUATORIAL_RADIUS - Constants.IAU_2015_NOMINAL_EARTH_POLAR_RADIUS)/Constants.IAU_2015_NOMINAL_EARTH_POLAR_RADIUS
        else:
            self.mu = Constants.WGS84_EARTH_MU
            self.R_earth = Constants.WGS84_EARTH_EQUATORIAL_RADIUS
            self.f_earth = Constants.WGS84_EARTH_FLATTENING

        self.M_earth = self.mu/scipy.constants.G

        self.__params = None

        self.inputFrame = self._get_frame(self.settings['in_frame'])
        self.outputFrame = self._get_frame(self.settings['out_frame'])

        if self.inputFrame.isPseudoInertial():
            self.inertialFrame = self.inputFrame
        else:
            self.inertialFrame = FramesFactory.getEME2000()

        self.body = OneAxisEllipsoid(self.R_earth, self.f_earth, self.outputFrame)

        self._forces = {}

        if self.settings['radiation_pressure']:
            self._forces['radiation_pressure'] = None
        if self.settings['drag_force']:
            self._forces['drag_force'] = None

        if self.settings['earth_gravity'] == 'HolmesFeatherstone':
            provider = org.orekit.forces.gravity.potential.GravityFieldFactory.getNormalizedProvider(self.gravity_order[0], self.gravity_order[1])
            holmesFeatherstone = org.orekit.forces.gravity.HolmesFeatherstoneAttractionModel(
                FramesFactory.getITRF(IERSConventions.IERS_2010, True),
                provider,
            )
            self._forces['earth_gravity'] = holmesFeatherstone

        elif self.settings['earth_gravity'] == 'Newtonian':
            Newtonian = org.orekit.forces.gravity.NewtonianAttraction(self.mu)
            self._forces['earth_gravity'] = Newtonian

        else:
            raise Exception('Supplied Earth gravity model "{}" not recognized'.format(self.settings['earth_gravity']))

        if self.settings['solarsystem_perturbers'] is not None:
            for body in self.settings['solarsystem_perturbers']:
                body_template = getattr(CelestialBodyFactory, 'get{}'.format(body))
                body_instance = body_template()
                perturbation = org.orekit.forces.gravity.ThirdBodyAttraction(body_instance)

                self._forces['perturbation_{}'.format(body)] = perturbation

    def __str__(self):
        '''
        ret = ''
        ret += 'PropagatorOrekit instance @ {}:'.format(hash(self)) + '\n' + '-'*25 + '\n'
        ret += '{:20s}: '.format('Integrator') + self.integrator + '\n'
        ret += '{:20s}: '.format('Minimum step') + str(self.minStep) + ' s' + '\n'
        ret += '{:20s}: '.format('Maximum step') + str(self.maxStep) + ' s' + '\n'
        ret += '{:20s}: '.format('Position Tolerance') + str(self.position_tolerance) + ' m' + '\n'
        if self._tolerances is not None:
            ret += '{:20s}: '.format('Absolute Tolerance') + str(JArray_double.cast_(self._tolerances[0])) + ' m' + '\n'
            ret += '{:20s}: '.format('Relative Tolerance') + str(JArray_double.cast_(self._tolerances[1])) + ' m' + '\n'
        ret += '\n'
        ret += '{:20s}: '.format('Input frame') + self.in_frame + '\n'
        ret += '{:20s}: '.format('Output frame') + self.out_frame + '\n'
        ret += '{:20s}: '.format('Gravity model') + self.earth_gravity + '\n'
        if self.earth_gravity == 'HolmesFeatherstone':
            ret += '{:20s} - Harmonic expansion order {}'.format('', self.gravity_order) + '\n'
        ret += '{:20s}: '.format('Atmosphere model') + self.atmosphere + '\n'
        ret += '{:20s}: '.format('Solar model') + self.solar_activity + '\n'
        ret += '{:20s}: '.format('Constants') + self.constants_source + '\n'
        ret += 'Included forces:' + '\n'
        for key in self._forces:
            ret += ' - {}'.format(' '.join(key.split('_'))) + '\n'
        ret += 'Third body perturbations:' + '\n'
        for body in self.solarsystem_perturbers:
            ret += ' - {:}'.format(body) + '\n'
        '''
        #return ret
        pass


    def _get_frame(self, name):
        '''Uses a string to identify which coordinate frame to initialize from Orekit package.

        See `Orekit FramesFactory <https://www.orekit.org/static/apidocs/org/orekit/frames/FramesFactory.html>`_
        '''

        if name == 'EME':
            return FramesFactory.getEME2000()
        elif name == 'CIRF':
            return FramesFactory.getCIRF(IERSConventions.IERS_2010, not self.frame_tidal_effects)
        elif name == 'ITRF':
            return FramesFactory.getITRF(IERSConventions.IERS_2010, not self.frame_tidal_effects)
        elif name == 'TIRF':
            return FramesFactory.getTIRF(IERSConventions.IERS_2010, not self.frame_tidal_effects)
        elif name == 'ITRFEquinox':
            return FramesFactory.getITRFEquinox(IERSConventions.IERS_2010, not self.frame_tidal_effects)
        if name == 'TEME':
            return FramesFactory.getTEME()
        else:
            raise Exception('Frame "{}" not recognized'.format(name))


    def _construct_propagator(self, initialOrbit):
        '''
        Get the specified integrator from hipparchus package. List available at: `nonstiff ode <https://www.hipparchus.org/apidocs/org/hipparchus/ode/nonstiff/package-summary.html>`_

        Configure the integrator tolerances using the orbit.
        '''
        self._tolerances = NumericalPropagator.tolerances(
                self.position_tolerance,
                initialOrbit,
                initialOrbit.getType()
            )

        integrator_constructor = getattr(
            org.hipparchus.ode.nonstiff,
            '{}Integrator'.format(self.settings['integrator']),
        )

        integrator = integrator_constructor(
            self.minStep,
            self.maxStep, 
            JArray_double.cast_(self._tolerances[0]),
            JArray_double.cast_(self._tolerances[1]),
        )

        propagator = NumericalPropagator(integrator)
        propagator.setOrbitType(initialOrbit.getType())

        self.propagator = propagator


    def _set_forces(self, A, cd, cr):
        '''Using the spacecraft specific parameters, set the drag force and radiation pressure models.
        
        **See:**
            * `drag <https://www.orekit.org/static/apidocs/org/orekit/forces/drag/package-summary.html>`_
            * `radiation <https://www.orekit.org/static/apidocs/org/orekit/forces/radiation/package-summary.html>`_
        '''
        __params = [A, cd, cr]

        re_calc = True
        if self.__params is None:
            re_calc = True
        else:
            if not np.allclose(np.array(__params,dtype=np.float), self.__params, rtol=1e-3):
                re_calc = True

        if re_calc:        
            self.__params = __params
            if self.settings['drag_force']:
                if self.settings['solar_activity'] == 'Marshall':
                    msafe = org.orekit.forces.drag.atmosphere.data.MarshallSolarActivityFutureEstimation(
                        "(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\\p{Digit}\\p{Digit}\\p{Digit}\\p{Digit}F10\\.(?:txt|TXT)",
                        self.SolarStrengthLevel,
                    )
                    manager = org.orekit.data.DataProvidersManager.getInstance()
                    manager.feed(msafe.getSupportedNames(), msafe)

                    if self.settings['atmosphere'] == 'DTM2000':
                        atmosphere_instance = org.orekit.forces.drag.atmosphere.DTM2000(msafe, CelestialBodyFactory.getSun(), self.body)
                    else:
                        raise Exception('Atmosphere model not recognized')

                    drag_model = org.orekit.forces.drag.DragForce(
                        atmosphere_instance,
                        org.orekit.forces.drag.IsotropicDrag(float(A), float(cd)),
                    )

                    self._forces['drag_force'] = drag_model
                else:
                    raise Exception('Solar activity model not recognized')

            if self.settings['radiation_pressure']:
                radiation_pressure_model = org.orekit.forces.radiation.SolarRadiationPressure(
                    CelestialBodyFactory.getSun(),
                    self.body.getEquatorialRadius(),
                    org.orekit.forces.radiation.IsotropicRadiationSingleCoefficient(float(A), float(cr)),
                )

                self._forces['radiation_pressure'] = radiation_pressure_model

            #self.propagator.removeForceModels()

            for force_name, force in self._forces.items():
                self.propagator.addForceModel(force)

        
    def propagate(self,t,state0,mjd0, **kwargs):
        '''
        **Implementation:**
    
        Units are in meters and degrees.

        Keyword arguments are:

            * float A: Area in m^2
            * float C_D: Drag coefficient
            * float C_R: Radiation pressure coefficient
            * float m: Mass of object in kg

        *NOTE:*
            * If the eccentricity is below 1e-10 the eccentricity will be set to 1e-10 to prevent Keplerian Jacobian becoming singular.
        

        The implementation first checks if the input frame is Pseudo inertial, if this is true this is used as the propagation frame. If not it is automatically converted to EME (ECI-J2000).

        Since there are forces that are dependent on the space-craft parameters, if these parameter has been changed since the last iteration the numerical integrator is re-initialized at every call of this method. The forces that can be initialized without spacecraft parameters (e.g. Earth gravitational field) are done at propagator construction.

        **Uses:**
            * :func:`propagator_base.PropagatorBase._make_numpy`
            * :func:`propagator_orekit.PropagatorOrekit._construct_propagator`
            * :func:`propagator_orekit.PropagatorOrekit._set_forces`
            * :func:`dpt_tools.kep2cart`
            * :func:`dpt_tools.cart2kep`
            * :func:`dpt_tools.true2mean`
            * :func:`dpt_tools.mean2true`
            
        
        See :func:`propagator_base.PropagatorBase.get_orbit`.
        '''


        if self.settings['radiation_pressure']:
            if 'C_R' not in kwargs:
                raise Exception('Radiation pressure force enabled but no coefficient "C_R" given')
        else:
            kwargs['C_R'] = 1.0

        if self.settings['drag_force']:
            if 'C_D' not in kwargs:
                raise Exception('Drag force enabled but no drag coefficient "C_D" given')
        else:
            kwargs['C_D'] = 1.0

        t = self._make_numpy(t)

        initialDate = mjd2absdate(mjd0)

        pos = org.hipparchus.geometry.euclidean.threed.Vector3D(float(state0[0]), float(state0[1]), float(state0[2]))
        vel = org.hipparchus.geometry.euclidean.threed.Vector3D(float(state0[3]), float(state0[4]), float(state0[5]))
        PV_state = PVCoordinates(pos, vel)

        if not self.inputFrame.isPseudoInertial():
            transform = self.inputFrame.getTransformTo(self.inertialFrame, initialDate)
            PV_state = transform.transformPVCoordinates(PV_state)

        initialOrbit = CartesianOrbit(
            PV_state,
            self.inertialFrame,
            initialDate,
            self.mu + float(scipy.constants.G*kwargs['m']),
        )

        self._construct_propagator(initialOrbit)
        self._set_forces(kwargs['A'], kwargs['C_D'], kwargs['C_R'])

        initialState = SpacecraftState(initialOrbit)

        self.propagator.setInitialState(initialState)

        tb_inds = t < 0.0
        t_back = t[tb_inds]

        tf_indst = t >= 0.0
        t_forward = t[tf_indst]

        if len(t_forward) == 1:
            if np.any(t_forward == 0.0):
                t_back = t
                t_forward = []
                tb_inds = t <= 0

        state = np.empty((6, len(t)), dtype=np.float)
        step_handler = PropagatorOrekit.OrekitVariableStep()

        if len(t_back) > 0:
            _t = t_back
            _t_order = np.argsort(np.abs(_t))
            _t_res = np.argsort(_t_order)
            _t = _t[_t_order]
            _state = np.empty((6, len(_t)), dtype=np.float) 
            step_handler.set_params(_t, initialDate, _state, self.outputFrame)

            self.propagator.setMasterMode(step_handler)

            self.propagator.propagate(initialDate.shiftedBy(float(_t[-1])))
            
            #now _state is full and in the order of _t
            state[:, tb_inds] = _state[:, _t_res]

        if len(t_forward) > 0:
            _t = t_forward
            _t_order = np.argsort(np.abs(_t))
            _t_res = np.argsort(_t_order)
            _t = _t[_t_order]
            _state = np.empty((6, len(_t)), dtype=np.float) 
            step_handler.set_params(_t, initialDate, _state, self.outputFrame)

            self.propagator.setMasterMode(step_handler)

            self.propagator.propagate(initialDate.shiftedBy(float(_t[-1])))
            
            #now _state is full and in the order of _t
            state[:, tf_indst] = _state[:, _t_res]

        return state
