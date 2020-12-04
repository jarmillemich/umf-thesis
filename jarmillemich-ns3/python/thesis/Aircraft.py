from math import pi
from sage.all import *
from .wings import loadWings
import os

package_directory = os.path.dirname(os.path.abspath(__file__))

# Constants
# Conversion factor
deg2rad = pi / 180
    
# Technically functions of altitude
g = 9.8
airDensity = 1.0

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
                 airfoil = 'RG15-Re0.2'):
        
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
        
        
    def loadWing(self, airfoil):
        self.airfoil = airfoil
        filePath = os.path.join(package_directory, 'contrib', 'wings', airfoil + '.csv')
        self.Cl, self.Cd, self.Cm = loadWings(filePath)
        
        # Also generate fitted functions so we can operate symbolically
        self.ClFit = makeFit(self.Cl)
        self.CdFit = makeFit(self.Cd)
        
        α, v = var('α, v')
        
        q = airDensity * v**2 / 2
        self.L = (self.ClFit * q * self._wingSurface).function(α,v)
        self.D_p = (self.CdFit * q * self._wingSurface).function(α,v)
        self.D_i = (self.L**2 / (pi * self._e0 * self._aspectRatio * self._wingSurface * q)).function(α,v)
        self.D = (self.D_p + self.D_i).function(α,v)
    
    ##########################################
    ########### Common functions #############
    ##########################################
    def powerFunctions(self, α = var('α')):
        # Coefficients dependent on angle of attack
        L0 = self.ClFit(α=α)
        D0 = self.CdFit(α=α) + self.ClFit(α=α)**2 / self.k2
        
        return L0, D0
    
    
    ##########################################
    ########### Straight path stuff ##########
    ##########################################
    
    # Needed velocity for different ascent/descent angles and angles of attack
    # Assuming thrust vector is in-line with angle of attack
    # theta is in radians, alpha is in degrees (TODO make consistent?)
    # See (some appendix) for the derivation
    def straightVelocity(self, θ = var('θ'), α = var('α'), a = var('a')):
        # Net force
        F = self._mass * a
        
        L0, D0 = self.powerFunctions(α)
        
        # Items to compute our velocity function
        num = cot(θ + α * deg2rad) * (F * sin(θ) + self._mass * g) - F * cos(θ)
        den = self.k1 * ( \
            L0 * sin(θ) + \
            D0 * cos(θ) + \
            cot(θ + α * deg2rad) * ( \
                L0 * cos(θ) - D0 * sin(θ) \
            ) \
        )

        # The final velocity function
        return sqrt(num / den)
        
    # Thrust needed for some θ, α, a
    def straightThrust(self, θ = var('θ'), α = var('α'), a = var('a')):
        # Net force
        F = self._mass * a
        v = self.straightVelocity(θ, α, a)
        thr = (F * cos(θ) + self.L(α=α, v=v) * sin(θ) + self.D(α=α, v=v) * cos(θ)) / cos(θ + α * deg2rad)
        # TODO fix this elsewhere so it doesn't happen (problem constraint? input formulation?)
        # if thr < 0:
        #     print('Warning, got a negative thrust')
        return thr
    
    # Power use for some θ, α, a
    def straightPower(self, θ = var('θ'), α = var('α'), a = var('a')):
        return self.straightVelocity(θ, α, a) * self.straightThrust(θ, α, a)
    
    def fastStraightVelocityThrustPower(self, θ, α, a):
        # Just use regular mathematical functions
        from math import sin, cos, tan, sqrt
        
        # Net force
        F = self._mass * a
        
#         L0, D0 = self.powerFunctions(α)
#         L0, D0 = L0.n(), D0.n()
        L0 = self.ClFit(α=α).n()
        D0 = self.CdFit(α=α).n() + L0**2 / self.k2
        
        # Items to compute our velocity function
        cotanThetaAlpha = (1 / tan(θ + α * deg2rad))
        num = cotanThetaAlpha * (F * sin(θ) + self._mass * g) - F * cos(θ)
        den = self.k1 * ( \
            L0 * sin(θ) + \
            D0 * cos(θ) + \
            cotanThetaAlpha * ( \
                L0 * cos(θ) - D0 * sin(θ) \
            ) \
        )
        
        v = sqrt(num / den)
        
        # Thrust function
        thr = (F * cos(θ) + self.L(α=α, v=v) * sin(θ) + self.D(α=α, v=v) * cos(θ)) / cos(θ + α * deg2rad)
        
        # Power function
        p = v * thr
        
        return v, thr, p
    
    ##########################################
    ########### Arced path stuff #############
    ##########################################
    
    # The roll angle needed for some radius given and angle of attack
    def turningRoll(self, r = var('r'), α = var('α')):
        L0, D0 = self.powerFunctions(α)
        
        denom = r * self.k1 * (D0(α=α) * tan(α * deg2rad) - L0(α=α))
        phi = asin(self._mass / denom)
        
        return phi
    
    # Needed velocity for turning
    # Assuming constant velocity, altitude, radius
    def turningVelocity(self, r = var('r'), α = var('α')):
        roll = self.turningRoll(r=r, α=α)
        L0, D0 = self.powerFunctions(α)
        
        denomPart = D0(α=α) * tan(α * deg2rad) * sin(roll) + L0(α=α) * cos(roll)
        
        vSquared = self._mass * g / (self.k1 * denomPart)
        
        return sqrt(vSquared)
    
    # Needed thrust for turning
    def turningThrust(self, r = var('r'), α = var('α')):
        L0, D0 = self.powerFunctions(α)
        return self.k1 * self.turningVelocity(r, α) ** 2 * D0(α=α) / cos(α * deg2rad)
    
    def turningPower(self, r = var('r'), α = var('α')):
        return self.turningVelocity(r, α) * self.turningThrust(r, α)
    
    def fastTurningVelocityThrustPower(self, r, α):
        # Just use regular mathematical functions
        from math import sin, cos, tan, sqrt, asin

        #         L0, D0 = self.powerFunctions(α)
        #         L0, D0 = L0.n(), D0.n()
        L0 = self.ClFit(α=α).n()
        D0 = self.CdFit(α=α).n() + L0**2 / self.k2
        
        # Roll
        denom = r * self.k1 * (D0 * tan(α * deg2rad) - L0)
        roll = asin(self._mass / denom)
        
        # Velocity
        denomPart = D0 * tan(α * deg2rad) * sin(roll) + L0 * cos(roll)
        
        vSquared = self._mass * g / (self.k1 * denomPart)
        
        v = sqrt(vSquared)
        
        # Thrust
        thr = self.k1 * vSquared * D0 / cos(α * deg2rad)
        
        p = thr * v
        
        return v, thr, p