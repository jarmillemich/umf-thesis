from math import pi
from sage.all import *
from .wings import loadWings
import os
from math import inf

package_directory = os.path.dirname(os.path.abspath(__file__))

# Constants
# Conversion factor
deg2rad = pi / 180

useSimpleAltitudeModel = False
    
if useSimpleAltitudeModel:
    # Use values at approximately 2 km
    h = var('h')
    g = 9.8 + 0 * h
    airDensity = 1.0 + 0 * h
else:
    # From https://en.wikipedia.org/wiki/Gravity_of_Earth and https://www.engineeringtoolbox.com/standard-atmosphere-d_604.html
    h = var('h')
    earth_mean_radius = 6371008.8 # meters
    g = 9.807 * (earth_mean_radius / (earth_mean_radius + h))**2

    gf = fast_callable(g, vars=[h])

    # From https://en.wikipedia.org/wiki/Density_of_air#:~:text=The%20density%20of%20air%20or,atmospheric%20pressure%2C%20temperature%20and%20humidity.
    # Good up to no more than 18 km (validated up to 10 km against the engineering toolbox data, off < 1 g / m^3)
    temp_sea_level = 288.15 # K
    temp_lapse = 0.0065 # K/m
    temp = temp_sea_level - temp_lapse * h # K
    pressure_sea_level = 101325 # Pa
    air_molar_mass = 0.0289654 # kg / mol
    gas_constant = 8.31447 # J/(mol*K)
    temp_exp = g * air_molar_mass / (gas_constant * temp_lapse)
    abs_pressure = pressure_sea_level * (temp / temp_sea_level)**temp_exp
    airDensity = abs_pressure * air_molar_mass / (gas_constant * temp)

def makeFit(inFunc):
    # We have to sort of make a function out of Cl and Cd...
    # It only has to be good in the domain so a regression should be fine
    a, b, c, d, e, f, alpha = var('a, b, c, d, e, f, α')
    model = a * alpha**5 + b * alpha**4 + c * alpha**3 + d * alpha**2 + e * alpha + f
    data = [(i, inFunc(i)) for i in xsrange(-8,12,0.1)]
    
    sol = find_fit(data, model, parameters=[a,b,c,d,e,f], variables=[alpha])
    return model(a=sol[0].rhs(), b = sol[1].rhs(), c = sol[2].rhs(), d = sol[3].rhs(), e = sol[4].rhs(), f = sol[5].rhs())

def makeFit2(inFunc):
    # We have to sort of make a function out of Cl and Cd...
    # It only has to be good in the domain so a regression should be fine
    a, b, c, d, e, f, alpha = var('a, b, c, d, e, f, α')
    model = a * alpha**5 + b * alpha**4 + c * alpha**3 + d * alpha**2 + e * alpha + f
    data = [(i, inFunc(i)) for i in xsrange(-8,12,0.1)]
    
    sol = find_fit(data, model, parameters=[a,b,c,d,e,f], variables=[alpha])

    oa = sol[0].rhs().n()
    ob = sol[1].rhs().n()
    oc = sol[2].rhs().n()
    od = sol[3].rhs().n()
    oe = sol[4].rhs().n()
    of = sol[5].rhs().n()

    return lambda alpha: oa * alpha**5 + ob * alpha**4 + oc * alpha**3 + od * alpha**2 + oe * alpha + of


class Aircraft:
    def __init__(self,
                 mass,
                 # Chord length (meters)
                 chord = 1,
                 # Wing span (meters)
                 wingSpan = 10,
                 # Oswald efficiency
                 e0 = 0.9,
                 # Airfoil to use
                 airfoil = 'RG15-Re0.2',
                 ### These defaults are taken from Oettershagen2017Design table 1
                 # Solar module efficiency
                 eff_solar = 0.237,
                 # MPPT module efficiency
                 eff_mppt = 0.95,
                 # Efficiency loss due to wing camber
                 eff_camber = 0.97,
                 # Efficiency of propulsion system (motors, gear boxes, propellor)
                 eff_prop = 0.62,
                 # Battery charging efficiency
                 eff_bat_charging = 0.95,
                 # Battery discharge efficiency (TODO figure out why this can be > 1)
                 eff_bat_discharging = 1.03,
                 # Percent of the wing surface area covered by solar panels
                 solar_fill = 0.85

                 ):
        
        # Let's say these should be constant WRT this object
        self._mass = mass
        
        self._chord = chord
        self._wingSpan = wingSpan
        self._e0 = e0
        
        self._aspectRatio = wingSpan / chord
        # Surface area (square meters)
        self._wingSurface = wingSpan * chord
        
        # Some constants we pull out of our equations
        self.k1 = airDensity * self._wingSurface / 2
        self.k2 = pi * self._e0 * self._aspectRatio

        # Assuming 10 m/s, 10 degrees C: http://www.airfoiltools.com/calculator/reynoldsnumber?MReNumForm%5Bvel%5D=10&MReNumForm%5Bchord%5D=1&MReNumForm%5Bkvisc%5D=1.4207E-5&yt0=Calculate
        self._re = 10 * self._chord / 1.4207e-5

        self.loadWing(airfoil)

        # Load power parameters
        self._efficiency = {
            'solar': eff_solar,
            'mppt': eff_mppt,
            'camber': eff_camber,
            'prop': eff_prop,
            'bat_charging': eff_bat_charging,
            'bat_discharging': eff_bat_discharging
        }

        self._solarArea = self._wingSurface * solar_fill
        
        
    def loadWing(self, airfoil):
        self.airfoil = airfoil
        filePath = os.path.join(package_directory, 'contrib', 'wings', airfoil + '.csv')
        self.Cl, self.Cd, self.Cm = loadWings(filePath)
        
        # Also generate fitted functions so we can operate symbolically
        self.ClFit = makeFit(self.Cl)
        self.CdFit = makeFit(self.Cd)
        
        α, v, h = var('α, v, h')
        
        q = airDensity(h=h) * v**2 / 2
        # Lift force
        self.L = (self.ClFit * q * self._wingSurface).function(α,v,h)

        # Coefficients of drag
        # Wing
        CDW = self.CdFit
        # Parasitic
        CDP = 0.074 * self._re ** -0.2
        # Induced
        CDI = self.ClFit ** 2 / self.k2
        # Sum
        CDA = CDW + CDP + CDI
        
        # Drag force
        self.D = (CDA * self._wingSurface * q).function(α,v,h)


        # Our L1 and D1 variables
        self.L1 = (self.k1 * self.ClFit).function(α,h)
        self.D1 = (self.k1 * CDA).function(α,h)

        # Testing
        self.L1f = fast_callable(self.L1, vars=[α,h], domain=RDF)
        self.D1f = fast_callable(self.D1, vars=[α,h], domain=RDF)
    
    ##########################################
    ########### Common functions #############
    ##########################################
    
    # Calculate the solar irradiance values for the provided poses
    # poses should be a pd.Dataframe, indexed by Datetime, with both 'tilt', 'azimuth', and 'z' (altitude AMSL) Series
    def calcSolarIrradiance(self, pose, latitude, longitude):
        import pandas as pd
        import pvlib

        altitude = pose['z']
        
        tz, name = 'America/Detroit', 'Michigan'
        loc = pvlib.location.Location(latitude, longitude, tz, altitude, name)

        # Compute solar position parameters for each timeslot
        ephem_data = loc.get_solarposition(pose.index)
        # Compute irradiation data for each timeslot
        irrad_data = loc.get_clearsky(pose.index, solar_position=ephem_data)
        # Extraterrestrial radiation value
        dni_et = pvlib.irradiance.get_extra_radiation

        # Sun position
        sun_zenith = ephem_data['apparent_zenith']
        sun_azimuth = ephem_data['azimuth']

        # Air mass
        AM = pvlib.atmosphere.get_relative_airmass(sun_zenith)

        surf_tilt = pose['tilt']
        surf_azimuth = pose['azimuth']

        return pvlib.irradiance.get_total_irradiance(
            surf_tilt, surf_azimuth,
            sun_zenith, sun_azimuth,
            dni = irrad_data['dni'], ghi = irrad_data['ghi'], dhi = irrad_data['dhi'],
            dni_extra = dni_et,
            model = 'klucher'
        )

    # Calculates the solar power over time
    def calcSolarPower(self, pose, latitude, longitude):
        rad = self.calcSolarIrradiance(pose, latitude, longitude)
        # Oettershagen2017Design eq 11
        return rad['poa_global'] * self._solarArea * self._efficiency['solar'] * self._efficiency['mppt']
    
    def calcBatteryChargeOld(self, pose, solar, bat_capacity_Wh, initial_charge = 0.5, constant_draw = 0):
        import pandas as pd
        bat_Wh = [0 for t in solar.index]
        # TODO we assume this is constant...
        seconds = (solar.index[1] - solar.index[0]).total_seconds()
        
        # This is sort of a left-hand Riemann integration?
        # TODO this can be vectorized, probably
        bat_Wh[0] = bat_capacity_Wh * initial_charge
        for idx in range(1, len(solar)):
            t = solar.index[idx]
            P_out = pose.power[t] + constant_draw
            P_in = solar[t]
            P_bat = P_in - P_out
            if P_bat > 0:
                # Charging
                P_bat *= self._efficiency['bat_charging']
            else:
                # discharging
                P_bat *= self._efficiency['bat_discharging']
                
            # Ws to Wh
            P_bat /= 3600
                
            bat_Wh[idx] = bat_Wh[idx - 1] + seconds * P_bat
            if bat_Wh[idx] > bat_capacity_Wh:
                bat_Wh[idx] = bat_capacity_Wh

        
        #print('weep')
        return pd.Series(bat_Wh, index = solar.index, dtype=float)

    def calcBatteryCharge(self, poses, solar, bat_capacity_Wh, initial_charge = 0.5, constant_draw = 0):
        # Only about 500x faster than the for loop version
        import numpy as np
        import pandas as pd
        # TODO we assume this is constant...
        dt = (solar.index[1] - solar.index[0]).total_seconds()
        
        power = poses.power + constant_draw
        p = np.array(solar - power) * dt / 3600
        p[p > 0] *= self._efficiency['bat_charging']
        p[p < 0] *= self._efficiency['bat_discharging']

        e = np.cumsum(p) + initial_charge * bat_capacity_Wh - p[0]

        erem = e
        prem = p

        while any(erem > bat_capacity_Wh):
            idx1 = np.argmax(erem > bat_capacity_Wh)
            cut = np.argmax(prem[idx1:] < 0) + idx1
            # We don't come back down
            if cut == idx1: break
            erem = erem[cut:]
            prem = prem[cut:]
            erem[:] = np.cumsum(prem) + bat_capacity_Wh
            
        e[e > bat_capacity_Wh] = bat_capacity_Wh

        
        #print('weep')
        return pd.Series(e, index = solar.index, dtype=float)
            
    def calcExcessTime(self, battery, constant_draw = 0):
        import pandas as pd
        # Just use a constant-altitude flight
        coastPower = self.fastGeneralVelocityThrustPower(5)[2]

        # Take minimum SoC after some time has passed
        minBattery = battery[battery.index[0] + pd.to_timedelta('1h'):].min() * 3600

        # minimum SoC energy over constant altitude power use
        return minBattery / (coastPower + constant_draw)

    def calcChargeMargin(self, battery):
        import pandas as pd
        # Move over to the second day
        secondDay = battery[battery.index[0] + pd.to_timedelta('1day'):]

        # Gather our left and right edge
        slice1 = secondDay[(secondDay == secondDay.max()).argmax():]
        slice2 = slice1[:(slice1 < slice1.max()).argmax()]

        return (slice2.index[-1] - slice2.index[0]).total_seconds()
    
    ##########################################
    ########### Straight path stuff ##########
    ##########################################
    
    def fastStraightVelocityThrustPower(self, θ, α, h):
        # Just use regular mathematical functions
        from math import sin, cos, tan, sqrt
        
        
        L1 = self.L1(α=α, h=h).n()
        D1 = self.D1(α=α, h=h).n()
        
        # Items to compute our velocity function
        cotanThetaAlpha = (1 / tan(θ + α * deg2rad))
        num = self._mass * g(h=h) * cotanThetaAlpha
        den = L1 * sin(θ) + \
              D1 * cos(θ) + \
              cotanThetaAlpha * ( \
                  L1 * cos(θ) - D1 * sin(θ) \
              )

        v = sqrt(num / den)
        
        # Thrust function
        thr = (self.L(α=α, v=v, h=h).n() * sin(θ) + self.D(α=α, v=v, h=h).n() * cos(θ)) / cos(θ + α * deg2rad)

        # If we are diving or something, null out our thrust (we aren't a turbine)
        if thr < 0:
            thr = 0
        
        # Power function
        p = v * thr
        
        return v, thr, p
    
    ##########################################
    ########### Arced path stuff #############
    ##########################################
        
    def fastTurningVelocityThrustPower(self, r, α, h):
        # Just use regular mathematical functions
        from math import sin, cos, tan, sqrt, asin


        L1 = self.L1(α=α, h=h).n()
        D1 = self.D1(α=α, h=h).n()
        
        # Roll
        denomRoll = r * (D1 * tan(α * deg2rad) + L1)
        # if abs(self._mass) > abs(denomRoll):
        #     print('too tight a turn!', self._mass, denomRoll)
        roll = asin(self._mass / denomRoll)
        
        # Velocity
        denomVelocity = D1 * tan(α * deg2rad) * sin(roll) + L1 * cos(roll)
        
        vSquared = self._mass * g(h=h) / denomVelocity
        
        
        
        v = sqrt(vSquared)
        
        # Thrust
        thr = vSquared * D1 / cos(α * deg2rad)
        
        p = thr * v
        
        return v, thr, p

    def bankAngleToTurnRadius(self, alpha, bank, theta = 0, height = 1000):
        # Convert a bank angle to a turning radius
        # Just use regular mathematical functions
        from math import sin, cos, tan, sqrt, asin

        thetaAlpha = theta + alpha * deg2rad

        L1 = float(self.L1f(alpha, height))
        D1 = float(self.D1f(alpha, height))

        # Some shorthand
        st = sin(theta)
        ct = cos(theta)
        A = L1 * ct - D1 * st
        B = L1 * st + D1 * ct
        denomPart = A + tan(thetaAlpha) * B

        return self._mass / (sin(bank * deg2rad) * denomPart)

    def turnRadiusToBankAngle(self, alpha, radius, theta = 0, height = 1000):
        # Just use regular mathematical functions
        from math import sin, cos, tan, sqrt, asin, pi
        deg2rad = pi / 180

        thetaAlpha = theta + alpha * deg2rad

        L1 = float(self.L1f(alpha, height))
        D1 = float(self.D1f(alpha, height))

        # Some shorthand
        st = sin(theta)
        ct = cos(theta)
        A = L1 * ct - D1 * st
        B = L1 * st + D1 * ct
        denomPart = A + tan(thetaAlpha) * B

        return asin(self._mass / (radius * denomPart)) / deg2rad



    ##########################################
    ## General formulae that work for both ###
    ##########################################
    def fastGeneralVelocityThrustPower(self, alpha, theta = 0, radius = inf, height = 1000):
        # Just use regular mathematical functions
        from math import sin, cos, tan, sqrt, asin

        #L1 = self.L1(α=alpha, h=height).n()
        #D1 = self.D1(α=alpha, h=height).n()
        L1 = float(self.L1f(alpha, height))
        D1 = float(self.D1f(alpha, height))

        thetaAlpha = theta + alpha * deg2rad

        # Some shorthand
        A = L1 * cos(theta) - D1 * sin(theta)
        B = L1 * sin(theta) + D1 * cos(theta)

        denomPart = A + tan(thetaAlpha) * B

        if radius == inf:
            # Straight line
            vSquared = self._mass * gf(height) / denomPart
        else:
            # Circular
            # Slight discrepancy from the specific case, <1% so chalk it up to floats
            roll = asin(self._mass / (radius * denomPart))
            vSquared = radius * gf(height) * tan(roll)

        thr = vSquared * B / cos(thetaAlpha)

        # XXX
        if thr < 0:
            thr = 0

        # if vSquared < 0:
        #     print('too slow', vSquared, alpha, theta, radius, height)

        v = sqrt(vSquared)
        p = v * thr
        return v, thr, p
