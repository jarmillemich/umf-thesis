from math import pi
from sage.all import *
from .wings import loadWings
import os

package_directory = os.path.dirname(os.path.abspath(__file__))

# Constants
# Conversion factor
deg2rad = pi / 180

useSimpleAltitudeModel = True
    
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
    model = a * alpha**4 + b * alpha**5 + c * alpha**3 + d * alpha**2 + e * alpha + f
    data = [(i, inFunc(i)) for i in xsrange(-8,12,0.1)]
    
    sol = find_fit(data, model, parameters=[a,b,c,d,e,f], variables=[alpha])
    return model(a=sol[0].rhs(), b = sol[1].rhs(), c = sol[2].rhs(), d = sol[3].rhs(), e = sol[4].rhs(), f = sol[5].rhs())

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
        
        self.loadWing(airfoil)
        
        # Some constants we pull out of our equations
        self.k1 = airDensity * self._wingSurface / 2
        self.k2 = pi * self._e0 * self._aspectRatio

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
        self.L = (self.ClFit * q * self._wingSurface).function(α,v,h)
        self.D_w = (self.CdFit * q * self._wingSurface).function(α,v,h)
        # Assuming 10 m/s, 10 degrees C: http://www.airfoiltools.com/calculator/reynoldsnumber?MReNumForm%5Bvel%5D=10&MReNumForm%5Bchord%5D=1&MReNumForm%5Bkvisc%5D=1.4207E-5&yt0=Calculate
        re = 10 * self._chord / 1.4207e-5
        self.D_p = 0.074 * re ** -0.2 * self._wingSurface * q
        self.D_i = (self.L**2 / (pi * self._e0 * self._aspectRatio * self._wingSurface * q)).function(α,v,h)
        self.D = (self.D_p + self.D_i + self.D_w).function(α,v,h)
    
    ##########################################
    ########### Common functions #############
    ##########################################
    def powerFunctions(self, α = var('α')):
        # Coefficients dependent on angle of attack
        L0 = self.ClFit(α=α)
        # Assuming 10 m/s, 10 degrees C: http://www.airfoiltools.com/calculator/reynoldsnumber?MReNumForm%5Bvel%5D=10&MReNumForm%5Bchord%5D=1&MReNumForm%5Bkvisc%5D=1.4207E-5&yt0=Calculate
        re = 10 * self._chord / 1.4207e-5
        D0 = self.CdFit(α=α) + self.ClFit(α=α)**2 / self.k2 + 0.074 * re**-0.2

        print(D0, re, 0.074 * re**-0.2)
        
        return L0, D0
    
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
        irrad_data = loc.get_clearsky(pose.index)
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
    
    def calcBatteryCharge(self, pose, solar, bat_capacity_Wh, initial_charge = 0.5, constant_draw = 0):
        import pandas as pd
        bat_Wh = [0 for t in solar.index]
        # TODO we assume this is constant...
        seconds = (solar.index[1] - solar.index[0]).total_seconds()
        
        # This is sort of a left-hand Riemann integration?
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
                
        return pd.Series(bat_Wh, index = solar.index, dtype=float)
            

    
    ##########################################
    ########### Straight path stuff ##########
    ##########################################
    
    # Needed velocity for different ascent/descent angles and angles of attack
    # Assuming thrust vector is in-line with angle of attack
    # theta is in radians, alpha is in degrees (TODO make consistent?)
    # See (some appendix) for the derivation
    def straightVelocity(self, θ = var('θ'), α = var('α'), a = var('a'), h = 1000):
        # Net force
        F = self._mass * a
        
        L0, D0 = self.powerFunctions(α)

        k1 = self.k1(h=h)
        
        # Items to compute our velocity function
        num = cot(θ + α * deg2rad) * (F * sin(θ) + self._mass * g(h=h)) - F * cos(θ)
        den = k1 * ( \
            L0 * sin(θ) + \
            D0 * cos(θ) + \
            cot(θ + α * deg2rad) * ( \
                L0 * cos(θ) - D0 * sin(θ) \
            ) \
        )

        if den == 0: return 0

        # The final velocity function
        return sqrt(num / den)
        
    # Thrust needed for some θ, α, a
    def straightThrust(self, θ = var('θ'), α = var('α'), a = var('a'), h = 1000):
        # Net force
        F = self._mass * a
        v = self.straightVelocity(θ, α, a, h)
        thr = (F * cos(θ) + self.L(α=α, v=v, h=h) * sin(θ) + self.D(α=α, v=v, h=h) * cos(θ)) / cos(θ + α * deg2rad)
        # TODO fix this elsewhere so it doesn't happen (problem constraint? input formulation?)
        if thr < 0:
            #print('Warning, got a negative thrust')
            # In "reality" we might pitch up more and glide at 0 power and conserve velocity
            return 0
        return thr
    
    # Power use for some θ, α, a
    def straightPower(self, θ = var('θ'), α = var('α'), a = var('a'), h = 1000):
        return self.straightVelocity(θ, α, a, h) * self.straightThrust(θ, α, a, h)
    
    def fastStraightVelocityThrustPower(self, θ, α, a, h):
        # Just use regular mathematical functions
        from math import sin, cos, tan, sqrt
        
        # Net force
        F = self._mass * a
        
#         L0, D0 = self.powerFunctions(α)
#         L0, D0 = L0.n(), D0.n()
        # Assuming 10 m/s, 10 degrees C: http://www.airfoiltools.com/calculator/reynoldsnumber?MReNumForm%5Bvel%5D=10&MReNumForm%5Bchord%5D=1&MReNumForm%5Bkvisc%5D=1.4207E-5&yt0=Calculate
        re = 10 * self._chord / 1.4207e-5
        L0 = self.ClFit(α=α).n()
        D0 = self.CdFit(α=α).n() + L0**2 / self.k2 + 0.074 * re**-0.2
        
        # Items to compute our velocity function
        cotanThetaAlpha = (1 / tan(θ + α * deg2rad))
        num = cotanThetaAlpha * (F * sin(θ) + self._mass * g(h=h)) - F * cos(θ)
        den = self.k1(h=h) * ( \
            L0 * sin(θ) + \
            D0 * cos(θ) + \
            cotanThetaAlpha * ( \
                L0 * cos(θ) - D0 * sin(θ) \
            ) \
        )
        
        v = sqrt(num / den)
        
        # Thrust function
        thr = (F * cos(θ) + self.L(α=α, v=v, h=h) * sin(θ) + self.D(α=α, v=v, h=h) * cos(θ)) / cos(θ + α * deg2rad)
        
        # Power function
        p = v * thr
        
        return v, thr, p
    
    ##########################################
    ########### Arced path stuff #############
    ##########################################
    
    # The roll angle needed for some radius given and angle of attack
    def turningRoll(self, r = var('r'), α = var('α'), h = 1000):
        L0, D0 = self.powerFunctions(α)
        
        k1 = self.k1(h=h)
        denom = r * k1 * (D0(α=α) * tan(α * deg2rad) - L0(α=α))
        phi = asin(self._mass / denom)
        
        return phi
    
    # Needed velocity for turning
    # Assuming constant velocity, altitude, radius
    def turningVelocity(self, r = var('r'), α = var('α'), h = 1000):
        roll = self.turningRoll(r=r, α=α, h=h)
        L0, D0 = self.powerFunctions(α)
        
        denomPart = D0(α=α) * tan(α * deg2rad) * sin(roll) + L0(α=α) * cos(roll)
        
        k1 = self.k1(h=h)
        vSquared = self._mass * g(h=h) / (k1 * denomPart)
        
        return sqrt(vSquared)
    
    # Needed thrust for turning
    def turningThrust(self, r = var('r'), α = var('α'), h = 1000):
        L0, D0 = self.powerFunctions(α)
        k1 = self.k1(h=h)
        return k1 * self.turningVelocity(r, α, h) ** 2 * D0(α=α) / cos(α * deg2rad)
    
    def turningPower(self, r = var('r'), α = var('α'), h = 1000):
        return self.turningVelocity(r, α, h) * self.turningThrust(r, α, h)
    
    def fastTurningVelocityThrustPower(self, r, α, h):
        # Just use regular mathematical functions
        from math import sin, cos, tan, sqrt, asin

        #         L0, D0 = self.powerFunctions(α)
        #         L0, D0 = L0.n(), D0.n()
        # Assuming 10 m/s, 10 degrees C: http://www.airfoiltools.com/calculator/reynoldsnumber?MReNumForm%5Bvel%5D=10&MReNumForm%5Bchord%5D=1&MReNumForm%5Bkvisc%5D=1.4207E-5&yt0=Calculate
        re = 10 * self._chord / 1.4207e-5
        L0 = self.ClFit(α=α).n()
        D0 = self.CdFit(α=α).n() + L0**2 / self.k2 + 0.074 * re**-0.2
        
        # Roll
        denom = r * self.k1(h=h) * (D0 * tan(α * deg2rad) - L0)
        roll = asin(self._mass / denom)
        
        # Velocity
        denomPart = D0 * tan(α * deg2rad) * sin(roll) + L0 * cos(roll)
        
        vSquared = self._mass * g(h=h) / (self.k1(h=h) * denomPart)
        
        v = sqrt(vSquared)
        
        # Thrust
        thr = self.k1(h=h) * vSquared * D0 / cos(α * deg2rad)
        
        p = thr * v
        
        return v, thr, p