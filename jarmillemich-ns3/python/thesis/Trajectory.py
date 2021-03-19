# a class for representing our "Waycircle" trajectories
# along with supporting utilities (including sage utils)
from sage.all import (
    vector, sqrt, atan2, line, arc,
    cos, sin, n, minimize, find_root,
    show, plot, RealSet, var, pi,
    circle, point
)
#from math import sqrt, atan2
from math import inf


# Get tangent points for circles
# http://www.ambrsoft.com/TrigoCalc/Circles2/Circles2Tangent_.htm ...
def getTangentWithPoint(x0, y0, r0, x1, y1, r1, xp, yp):
    # In the symbology of that page
    a, b, c, d = x0, y0, x1, y1
    
    # These are for the first circle
    denom0 = (xp - a)**2 + (yp - b)**2
    if denom0 - r0**2 < 0:
        dist = sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
        print('Tangent overlap0!', (x0, y0, r0), (x1, y1, r1), dist)
        #print(x0, y0, r0, x1, y1, r1, xp, yp, denom0, r0)
        (circle((x0, y0), r0) + circle((x1, y1), r1) + point((xp, yp))).show()
    sqrtTerm0 = sqrt(denom0 - r0**2)
    
    
    xt1 = (r0**2 * (xp - a) + r0 * (yp - b) * sqrtTerm0) / denom0 + a
    xt2 = (r0**2 * (xp - a) - r0 * (yp - b) * sqrtTerm0) / denom0 + a
    
    yt1 = (r0**2 * (yp - b) - r0 * (xp - a) * sqrtTerm0) / denom0 + b
    yt2 = (r0**2 * (yp - b) + r0 * (xp - a) * sqrtTerm0) / denom0 + b
    
    # These are for the second circle
    denom1 = (xp - c)**2 + (yp - d)**2
    if denom1 - r1**2 < 0:
        print('Tangent overlap!')
        #print(x0, y0, r0, x1, y1, r1, xp, yp, denom1, r1)
    sqrtTerm1 = sqrt(denom1 - r1**2)
    
    
    xt3 = (r1**2 * (xp - c) + r1 * (yp - d) * sqrtTerm1) / denom1 + c
    xt4 = (r1**2 * (xp - c) - r1 * (yp - d) * sqrtTerm1) / denom1 + c
    
    yt3 = (r1**2 * (yp - d) - r1 * (xp - c) * sqrtTerm1) / denom1 + d
    yt4 = (r1**2 * (yp - d) + r1 * (xp - c) * sqrtTerm1) / denom1 + d
    
    # A pair of pairs of points, each sub-pair is a tangent line segment
    return ((xt1, yt1), (xt3, yt3)), ((xt2, yt2), (xt4, yt4))


def getOuterTangent(x0, y0, r0, x1, y1, r1):
    # In the symbology of that page
    a, b, c, d = x0, y0, x1, y1
    
    D = sqrt((c - a)**2 + (d - b)**2)
    
    xp = (c * r0 - a * r1) / (r0 - r1)
    yp = (d * r0 - b * r1) / (r0 - r1)
    
    return getTangentWithPoint(x0, y0, r0, x1, y1, r1, xp, yp)

def getInnerTangent(x0, y0, r0, x1, y1, r1):
    # In the symbology of that page
    a, b, c, d = x0, y0, x1, y1
    
    D = sqrt((c - a)**2 + (d - b)**2)
    
    xp = (c * r0 + a * r1) / (r0 + r1)
    yp = (d * r0 + b * r1) / (r0 + r1)
    
    return getTangentWithPoint(x0, y0, r0, x1, y1, r1, xp, yp)
    
    
##########################################
########### Trajectory Pieces ############
##########################################
    
class LineSegment:
    def __init__(self, x0, y0, z0, x1, y1, z1):
        self.x0 = n(x0)
        self.y0 = n(y0)
        self.z0 = n(z0)
        
        self.x1 = n(x1)
        self.y1 = n(y1)
        self.z1 = n(z1)
        
        # Calculate some stats
        self.length = n(sqrt((x1 - x0)**2 + (y1 - y0)**2 + (z1 - z0)**2))
        self.groundLength = n(sqrt((x1 - x0)**2 + (y1 - y0)**2))
        self.theta = n(atan2(self.z1 - self.z0, self.groundLength))
        
        
    def render(self, **kwargs):
        return line(((self.x0, self.y0, self.z0), (self.x1, self.y1, self.z1)), **kwargs)

    def renderTop(self, **kwargs):
        return line(((self.x0, self.y0), (self.x1, self.y1)), **kwargs)

    def renderSide(self, xy = 'x', **kwargs):
        if xy == 'x':
            return line(((self.x0, self.z0), (self.x1, self.z1)), **kwargs)
        else:
            return line(((self.y0, self.z0), (self.y1, self.z1)), **kwargs)
    
    def piece(self, t0, v):
        dt = self.length / v
        
        area = RealSet.closed_open(t0, n(t0 + dt))
        t = var('t')
        time_at = t - t0
        #dx = (n(self.v0) * time_at + 1/2 * n(self.a) * time_at^2) / n(self.length)
        dx = time_at / dt
        
        p0 = vector((self.x0, self.y0, self.z0))
        p1 = vector((self.x1, self.y1, self.z1))
        dp = p1 - p0
        
        p = n(p0) + n(dp) * dx
        
        return area, p
    
    def fastPiece(self, t0, v):
        dt = self.length / v
        
        dx = self.x1 - self.x0
        dy = self.y1 - self.y0
        dz = self.z1 - self.z0
        
        return lambda t: (
            self.x0 + dx * (t - t0) / dt,
            self.y0 + dy * (t - t0) / dt,
            self.z0 + dz * (t - t0) / dt,
        )
    
    
    def toSimCode(self, craft, alpha):
        v, t, p = self.velocityThrustPower(craft, alpha)
        return 'enbMobility->addLineSegment(Vector(%.2f, %.2f, %.2f), Vector(%.2f, %.2f, %.2f), %.2f);' % (
            self.x0, self.y0, self.z0,
            self.y1, self.y1, self.z1,
            v
        )
    
    def toSim(self, craft, alpha):
        from ns.core import Vector
        from ns.mobility import PathMobilityModelSegments
        
        v, t, p = self.velocityThrustPower(craft, alpha)
        
        return PathMobilityModelSegments.LineSegment(
            Vector(self.x0, self.y0, self.z0),
            Vector(self.x1, self.y1, self.z1),
            v
        )
    

    # Get the optimal (alpha, power) for a given craft
    def optimalAlpha(self, craft):
        powerFun = craft.straightPower(θ = self.theta, a = 0)
        alpha = minimize(powerFun, [2])[0]
        
        pwr = n(powerFun(α=alpha))
        if pwr < 0:
            print(self.z0, self.z1, self.theta, alpha, pwr)
            # Likely we are descending
            # Find a 0-power alpha instead
            # TODO this might cause what those who know call a "stall", investigate
            show(plot(powerFun, 0, 15))
            alpha = find_root(powerFun, 0, 15)
        
        return alpha
    
    def velocityThrustPower(self, craft, alpha):
        # Take our average altitude/height
        h = (self.z0 + self.z1) / 2

        return craft.fastStraightVelocityThrustPower(θ = self.theta, α=alpha, h=h)

    def toPoses(self, times, t0, v, alpha, thrust = 0, power = 0, df = None):
        import pandas as pd
        import numpy as np
        import math
        dt = self.length / v

        dx = self.x1 - self.x0
        dy = self.y1 - self.y0
        dz = self.z1 - self.z0

        # Apparently can't have fractional times, so do milliseconds
        endTime = t0 + pd.offsets.Milli(int(dt * 1e3))

        # We are assuming that the solar panels are in-line with the wings
        # North is +Y
        # Azimuth is degrees east of north
        rad2deg = 180 / math.pi
        azimuth = 90 - math.atan2(dy, dx) * rad2deg
        length_horizontal = math.sqrt(dx*dx + dy*dy)
        tilt = -(math.atan2(dz, length_horizontal) * rad2deg + alpha)
        #print('heading', dx, dy, 'angle is', azimuth, 'going up', dz, alpha, 'for', tilt)

        slices = times[t0:endTime]
        dices = np.array([t.total_seconds() for t in (slices - t0)])
        #print(dt, len(slices), slices)
        ret = pd.DataFrame({
            'x': self.x0 + dx * dices / dt,
            'y': self.y0 + dy * dices / dt,
            'z': self.z0 + dz * dices / dt,
            # 'x': [self.x0 + dx * (t - t0).total_seconds() / dt for t in slices],
            # 'y': [self.y0 + dy * (t - t0).total_seconds() / dt for t in slices],
            # 'z': [self.z0 + dz * (t - t0).total_seconds() / dt for t in slices],
            #'z': [1000 for t in slices],
            'v': v,
            'tilt': tilt,
            'azimuth': azimuth % 360,
            'thrust': thrust,
            'power': power
        }, index=slices, dtype=float)

        # Extra millisecond to avoid overlap
        return endTime + pd.offsets.Milli(1), ret

    def toPosesTest(self, times, t_at, v, alpha, thrust = 0, power = 0, craft = None):
        import pandas as pd
        import numpy as np
        import math
        dt = (self.length / v).n()

        dx = self.x1 - self.x0
        dy = self.y1 - self.y0
        dz = self.z1 - self.z0


        # We are assuming that the solar panels are in-line with the wings
        # North is +Y
        # Azimuth is degrees east of north
        rad2deg = 180 / math.pi
        azimuth = 90 - math.atan2(dy, dx) * rad2deg
        length_horizontal = math.sqrt(dx*dx + dy*dy)
        tilt = -(math.atan2(dz, length_horizontal) * rad2deg + alpha)
        #print('heading', dx, dy, 'angle is', azimuth, 'going up', dz, alpha, 'for', tilt)

        
        dices = times[np.bitwise_and(times >= t_at, times < t_at + dt)] - t_at
        #print(dt, len(slices), slices)
        return dt, np.array([
            #slices,
            self.x0 + dx * dices / dt,
            self.y0 + dy * dices / dt,
            self.z0 + dz * dices / dt,
            # 'x': [self.x0 + dx * (t - t0).total_seconds() / dt for t in slices],
            # 'y': [self.y0 + dy * (t - t0).total_seconds() / dt for t in slices],
            # 'z': [self.z0 + dz * (t - t0).total_seconds() / dt for t in slices],
            #'z': [1000 for t in slices],
            np.full(len(dices), v),
            np.full(len(dices), tilt),
            np.full(len(dices), azimuth % 360),
            np.full(len(dices), thrust),
            np.full(len(dices), power),
            #alpha
            np.full(len(dices), alpha),
        ], dtype='float64')
        
class ArcSegment:
    def __init__(self, x, y, z, r, theta, dtheta):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.r = float(r)
        self.theta = float(theta)
        self.dtheta = float(dtheta)
        
        # Calculate some stats
        self.length = n(abs(self.r * self.dtheta))
        #self.dt = self.length / v
        
    def render(self, **kwargs):
        # Arc doesn't support plot3d, so we do this manually
        # Will definitely not support ascending/desccending arcs, yet
        x, y, r, s1, s2, z = self.x, self.y, self.r, self.theta, self.theta + self.dtheta, self.z
        n = 50
        dt = (s2 - s1) / n
        xdata = [x + r * cos(s1 + t * dt) for t in range(n + 1)]
        ydata = [y + r * sin(s1 + t * dt) for t in range(n + 1)]

        return line([
            (xdata[i], ydata[i], z)
            for i in range(n)
        ], **kwargs)

    def renderTop(self, **kwargs):
        return arc((self.x, self.y), self.r, self.r, 0, (self.theta, self.theta + self.dtheta), **kwargs)

    def renderSide(self, xy = 'x', **kwargs):
        # TODO actually do the circle projections
        x, y, r, s1, s2, z = self.x, self.y, self.r, self.theta, self.theta + self.dtheta, self.z
        n = 50
        dt = (s2 - s1) / n
        xdata = [x + r * cos(s1 + t * dt) for t in range(n + 1)]
        ydata = [y + r * sin(s1 + t * dt) for t in range(n + 1)]

        if xy == 'x':
            return line([
                (xdata[i], z)
                for i in range(n)
            ], **kwargs)
        else:
            return line([
                (ydata[i], z)
                for i in range(n)
            ], **kwargs)
    
    def piece(self, t0, v):
        dt = self.length / v
        area = RealSet.closed_open(t0, n(t0 + dt))
        
        t = var('t')
        time_at = t - t0
        dθdt = n(self.dtheta) / n(dt)
        
        theta_at = dθdt * time_at + n(self.theta)
        p0 = n(vector((self.x, self.y, self.z)))
        dp = vector((self.r * cos(theta_at), self.r * sin(theta_at), 0))
        p = p0 + dp
        
        return area, p
    
    def fastPiece(self, t0, v):
        from math import cos, sin
        
        dt = self.length / v
        dθdt = self.dtheta / dt
        
        return lambda t: (
            self.x + self.r * cos(dθdt * (t - t0) + self.theta),
            self.y + self.r * sin(dθdt * (t - t0) + self.theta),
            self.z
        )
    
    def toSimCode(self, craft, alpha):
        v, t, p = self.velocityThrustPower(craft, alpha)
        return 'enbMobility->addArcSegment(Vector(%.2f, %.2f, %.2f), %.2f, %.2f, %.2f, %.2f);' % (
            self.x, self.y, self.z,
            self.r, self.theta, self.dtheta,
            v
        )

    def toSim(self, craft, alpha):
        from ns.core import Vector
        from ns.mobility import PathMobilityModelSegments
        
        v, t, p = self.velocityThrustPower(craft, alpha)
        return PathMobilityModelSegments.ArcSegment(
            Vector(self.x, self.y, self.z),
            self.r, self.theta, self.dtheta,
            v
        )
    
    # Get the optimal (alpha, power) for a given craft
    def optimalAlpha(self, craft):
        powerFun = craft.turningPower(r = self.r)
        alpha = minimize(powerFun, [2])[0]
        return alpha
    
    # Gets the kinematic components of the flight given the craft and angle
    def velocityThrustPower(self, craft, alpha):
        #
        #         v = n(craft.turningVelocity(r = self.r, α=alpha))
        #         t = n(craft.turningThrust(r = self.r, α=alpha))
        #         p = n(craft.turningPower(r = self.r, α=alpha))
                
        #         return v, t, p
        return craft.fastTurningVelocityThrustPower(self.r, alpha, self.z)

    ## Experimental
    def toPoses(self, times, t0, v, alpha, thrust = 0, power = 0, df = None):
        import pandas as pd
        import numpy as np
        import math
        dt = self.length / v
        rate = self.dtheta / dt

        # Apparently can't have fractional times, so do milliseconds
        endTime = t0 + pd.offsets.Milli(int(dt * 1e3))

        # We are assuming that the solar panels are in-line with the wings
        # North is +Y
        # Azimuth is degrees east of north
        rad2deg = 180 / math.pi
        deg2rad = math.pi / 180
        azimuth = 0
        tilt = 0

        slices = times[t0:endTime]
        dices = np.array([t.total_seconds() for t in (slices - t0)])
        thetas = rate * dices + self.theta

        sign = 1 if self.dtheta > 0 else -1

        if df is None:
            ret = pd.DataFrame({
                'x': self.x + self.r * np.cos(thetas),
                'y': self.y + self.r * np.sin(thetas),
                'z': self.z,
                #'z': [1000 for t in slices],
                'v': v,
                'tilt': -alpha,
                # TODO this needs to be totally reworked to take into account our roll AND alpha
                #'azimuth': [(sign * (90 - math.atan2(math.cos(theta), -math.sin(theta)) * rad2deg)) % 360 for theta in thetas],
                'azimuth': (sign * (90 - np.arctan2(np.cos(thetas), -np.sin(thetas)) * rad2deg)) % 360,
                'thrust': thrust,
                'power': power
            }, index=slices, dtype=float)
        else:
            # Append instead
            ret = df

            mices = df.index.isin(slices)

            # df.loc[mices, 'x'] = self.x + self.r * np.cos(thetas)
            # df.loc[mices, 'y'] = self.y + self.r * np.sin(thetas)
            # df.loc[mices, 'z'] = self.z
            # df.loc[mices, 'v'] = v
            # df.loc[mices, 'tilt'] = -alpha
            # df.loc[mices, 'azimuth'] = (sign * (90 - np.arctan2(np.cos(thetas), -np.sin(thetas)) * rad2deg)) % 360
            # df.loc[mices, 'thrust'] = thrust
            # df.loc[mices, 'power'] = power
            df.assign

        # Extra millisecond to avoid overlap
        return endTime + pd.offsets.Milli(1), ret

    def toPosesTest(self, times, t_at, v, alpha, thrust = 0, power = 0, craft = None):
        import pandas as pd
        import numpy as np
        import math
        dt = self.length / v
        rate = self.dtheta / dt

        # We are assuming that the solar panels are in-line with the wings
        # North is +Y
        # Azimuth is degrees east of north
        rad2deg = 180 / math.pi
        deg2rad = math.pi / 180
        azimuth = 0
        tilt = 0

        dices = times[np.bitwise_and(times >= t_at, times < t_at + dt)] - t_at
        thetas = rate * dices + self.theta

        sign = 1 if self.dtheta > 0 else -1

        return dt, np.array([
            #slices,
            self.x + self.r * np.cos(thetas),
            self.y + self.r * np.sin(thetas),
            np.full(len(dices), self.z),
            #'z': [1000 for t in slices],
            np.full(len(dices), v),
            np.full(len(dices), -alpha),
            # TODO this needs to be totally reworked to take into account our roll AND alpha
            #'azimuth': [(sign * (90 - math.atan2(math.cos(theta), -math.sin(theta)) * rad2deg)) % 360 for theta in thetas],
            (sign * (90 - np.arctan2(np.cos(thetas), -np.sin(thetas)) * rad2deg)) % 360,
            np.full(len(dices), thrust),
            np.full(len(dices), power),
            #alpha
            np.full(len(dices), alpha)
        ], dtype='float64')


# Compute some things about an arc from p0 to p1 with the given radius
def pointToPointArc(p0, p1, radius):
    from math import sqrt, atan2, sin, cos
    x0, y0, z0 = [float(i) for i in p0]
    x1, y1, z1 = [float(i) for i in p1]
    radius = float(radius)
    dx = x1 - x0
    dy = y1 - y0

    directLength = sqrt(dx ** 2 + dy ** 2)
    mx = (x0 + x1) / 2
    my = (y0 + y1) / 2

    lm = sqrt((mx-x0)**2 + (my-y0)**2)

    # Angle from p0 to p1
    sTheta = atan2(my - y0, mx - x0)
    # Angle from midpoint to center
    # Let's say a negative radius will flip the circle from left hand to right hand
    flippy = (1 if radius >= 0 else -1)
    cTheta = sTheta + pi / 2
    # Distance to center
    if radius**2 - lm**2 < 0:
        print('P2PArc: Points are too close: radius=%.2f, lm=%.2f' % (radius, lm))
    cDist = flippy * sqrt(radius**2 - lm**2)

    cx = mx + cDist * cos(cTheta)
    cy = my + cDist * sin(cTheta)

    theta = atan2(y0 - cy, x0 - cx)
    dtheta = angleBetween((x0 - cx, y0 - cy), (x1 - cx, y1 - cy))

    return (cx, cy), (theta, dtheta)

# Rotate some vectors in R3
# x y z is right down forward
# roll is right, yaw is right, pitch is up
def rotate3(x, y, z, roll, yaw, pitch):
  # From https://en.wikipedia.org/wiki/Rotation_matrix#General_rotations
  # But vectorized
  from numpy import sin, cos
  ca = cos(roll)
  sa = sin(roll)
  cb = cos(yaw)
  sb = sin(yaw)
  cc = cos(pitch)
  sc = sin(pitch)
  
  xx = x * ca * cb + y * (ca * sb * sc - sa * cc) + z * (ca * sb * cc + sa * sc)
  yy = x * sa * cb + y * (sa * sb * sc + ca * cc) + z * (sa * sb * cc - ca * sc)
  zz = x * -sb + y * cb * sc + z * cb * cc
  
  return xx, yy, zz

# Rotate some vectors in R3
# Input xyz is forward, left, down
# order is pitch, roll, yaw
# all angles are CCW
# Output xyz is in east north down
def rotate4(x, y, z, yaw, roll, pitch):
  # From https://en.wikipedia.org/wiki/Rotation_matrix#General_rotations
  # But vectorized
  from numpy import sin, cos
  ca = cos(roll)
  sa = sin(roll)
  cb = cos(pitch)
  sb = sin(pitch)
  cc = cos(yaw)
  sc = sin(yaw)

  # Derived with this (Sagemath)
  # ============================
  # a, b, c = var('a b c')
  # # yaw
  # Rz = Matrix([[cos(c), -sin(c), 0], [sin(c), cos(c), 0], [0, 0, 1]])
  # # pitch
  # Ry = Matrix([[cos(b), 0, sin(b)], [0, 1, 0], [-sin(b), 0, cos(b)]])
  # # roll
  # Rx = Matrix([[1, 0, 0], [0, cos(a), -sin(a)], [0, sin(a), cos(a)]])
  # # c is yaw
  # # b is pitch
  # # a is roll
  # Yaw, roll, pitch

  
  xx = -x * (sa*sb*sc-cb*cc) + -y * (ca*sc) + z * (cb*sa*sc+cc*sb)
  yy = x * (cc*sa*sb+cb*sc) + y * (ca*cc) - z * (cb*cc*sa-sb*sc)
  zz = -x * (ca*sb) + y * (sa) + z * (ca*cb)
  
  return xx, yy, zz

class GeneralSegment:
    def __init__(self, p0, p1, radius = inf, loops = 0):
        from math import pi, sin, cos, atan2, sqrt

        sign = lambda x: x / abs(x) if x != 0 else 0
        signUp = lambda x: 1 if x >= 0 else -1

        if len(p0) != 3 or len(p1) != 3:
            raise TypeError('Please supply a 3-vec')
        
        self.x0, self.y0, self.z0 = self.p0 = [float(i) for i in p0]
        self.x1, self.y1, self.z1 = self.p1 = [float(i) for i in p1]
        self.radius = radius = float(radius)

        dx = self.dx = self.x1 - self.x0
        dy = self.dy = self.y1 - self.y0
        dz = self.dz = self.z1 - self.z0

        if radius == inf:
            # Line
            self.groundLength = sqrt(dx ** 2 + dy ** 2)
            self.length = sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        else:
            # Curve
            #self.groundLength = 2 * pi * self.radius
            directLength = sqrt(dx ** 2 + dy ** 2)

            # Nope

            #self.radius = radius = (self.radius + signUp(self.radius)) * directLength / 2
            #print('radius ends up ', self.radius)

            if abs(radius) * 2 < directLength:
                raise TypeError('Radius is too small %.2f < %.2f' % (abs(radius), directLength/2))

            
            

            # Find our center

            # Midpoint
            self.mx = mx = (self.x0 + self.x1) / 2
            self.my = my = (self.y0 + self.y1) / 2
            # Distance to midpoint
            lm = sqrt((mx-self.x0)**2 + (my-self.y0)**2)

            #print('mid and dist', mx, my, lm)

            # Angle from p0 to p1
            sTheta = atan2(my - self.y0, mx - self.x0)
            # Angle from midpoint to center
            # Let's say a negative radius will flip the circle from left hand to right hand
            flippy = (1 if radius >= 0 else -1)
            cTheta = sTheta + pi / 2
            # Distance to center
            if radius**2 - lm**2 < 0:
                print('Radius is too small:', radius, lm)
            cDist = flippy * sqrt(radius**2 - lm**2)

            #print('to mid, to center, branch length', sTheta, cTheta, cDist)

            # Center coordinates
            self.cx = mx + cDist * cos(cTheta)
            self.cy = my + cDist * sin(cTheta)

            #print('center is', self.cx, self.cy)

            #print('  and back is ', self.y0 - self.cy, self.x0 - self.cx)
            self.theta = atan2(self.y0 - self.cy, self.x0 - self.cx)
            if radius < 0: self.theta = pi + self.theta
            self.dtheta = angleBetween((self.x0 - self.cx, self.y0 - self.cy), (self.x1 - self.cx, self.y1 - self.cy))

            #print('start theta, delta theta', self.theta, self.dtheta)
            # Unsure if we really need this
            self.dtheta += 2 * pi * loops * flippy

            #self.dtheta *= flippy

            self.groundLength = abs(self.radius * self.dtheta)
            # good visualization here: https://math.stackexchange.com/questions/2160851/finding-the-length-of-a-helix
            self.length = sqrt(dz ** 2 + self.groundLength ** 2)

            #print('length (ground, total)', self.groundLength, self.length)

        # TODO I think this still works for circular...
        self.ascentAngle = atan2(self.z1 - self.z0, self.groundLength)
    
    def _renderPieces(self, n = 50):
        import numpy as np
        dices = np.arange(n+1) / n
        
        if self.radius == inf:
            dxdt = self.dx
            dydt = self.dy

            x = self.x0 + dices * dxdt
            y = self.y0 + dices * dydt
            z = self.z0 + dices * dzdt
        else:
            thetas = self.theta + dices * self.dtheta
            x = self.cx + self.radius * np.cos(thetas)
            y = self.cy + self.radius * np.sin(thetas)

            
        
        dzdt = self.dz
        z = self.z0 + dices * dzdt

        return x.tolist(), y.tolist(), z.tolist()



        # x, y, r, s1, s2, z = self.cx, self.cy, self.radius, self.theta, self.theta + self.dtheta, self.z0
        # dt = (s2 - s1) / n
        # xdata = [x + r * cos(s1 + t * dt) for t in range(n + 1)]
        # ydata = [y + r * sin(s1 + t * dt) for t in range(n + 1)]

        # dzdn = self.dz / n
        # zdata = [z + dzdn * t for t in range(n + 1)]

        # return xdata, ydata, zdata

    def render(self, n = 50, **kwargs):
        xdata, ydata, zdata = self._renderPieces(n)

        return line([
            (xdata[i], ydata[i], zdata[i])
            for i in range(len(xdata))
        ], **kwargs)

    def renderTop(self, n = 50, **kwargs):
        #return arc((self.x0, self.y0), self.radius, self.radius, 0, (self.theta, self.theta + self.dtheta), **kwargs)
        xdata, ydata, zdata = self._renderPieces(n)

        return line([
            (xdata[i], ydata[i])
            for i in range(len(xdata))
        ], **kwargs)

    def renderSide(self, xy = 'x', n = 50, **kwargs):
        xdata, ydata, zdata = self._renderPieces(n)

        if xy == 'x':
            return line([
                (xdata[i], zdata[i])
                for i in range(len(xdata))
            ], **kwargs)
        else:
            return line([
                (ydata[i], zdata[i])
                for i in range(len(xdata))
            ], **kwargs)

    def toSim(self, craft, alpha):
        from ns.core import Vector
        from ns.mobility import PathMobilityModelSegments
        
        v, t, p = self.velocityThrustPower(craft, alpha)

        if self.radius != inf:
            return PathMobilityModelSegments.GeneralSegment(
                Vector(self.cx, self.cy, self.z0),
                self.radius, self.theta, self.dtheta,
                self.dz, v
            )
        else:
            return PathMobilityModelSegments.GeneralSegment(
                Vector(self.x0, self.y0, self.z0),
                Vector(self.x1, self.y1, self.z1),
                v
            )
    
    # Gets the kinematic components of the flight given the craft and angle
    def velocityThrustPower(self, craft, alpha):
        # Just take mean altitude (appx)
        mz = (self.z0 + self.z1) / 2
        return craft.fastGeneralVelocityThrustPower(alpha, theta=self.ascentAngle, radius=self.radius, height=mz)

    def toPosesTest(self, times, t_at, v, alpha, thrust = 0, power = 0, craft = None):
        import pandas as pd
        import numpy as np
        import math
        dt = self.length / v
        

        # We are assuming that the solar panels are in-line with the wings
        # North is +Y
        # Azimuth is degrees east of north
        rad2deg = 180 / math.pi
        deg2rad = math.pi / 180
        azimuth = 0
        tilt = 0

        dices = times[np.bitwise_and(times >= t_at, times < t_at + dt)] - t_at

        if dt == 0:
            #print('something went wrong', self.length, v, self.dz, self.p0, self.p1)
            # Oh well
            return dt, np.array([[]]*9)

        dzdt = self.dz / dt

        

        if self.radius == inf:
            dx, dy, dz = self.dx, self.dy, self.dz
            
            dxdt = dx / dt
            dydt = dy / dt

            rad2deg = 180 / math.pi
            azimuth = 90 - math.atan2(dy, dx) * rad2deg
            length_horizontal = math.sqrt(dx*dx + dy*dy)
            tilt = -(math.atan2(dz, length_horizontal) * rad2deg + alpha)
            
            return dt, np.array([
                # X Y Z V
                self.x0 + dices * dxdt,
                self.y0 + dices * dydt,
                self.z0 + dices * dzdt,
                np.full(len(dices), v),
                # Tilt Azimuth
                np.full(len(dices), tilt),
                np.full(len(dices), azimuth % 360),
                # Thrust power
                np.full(len(dices), thrust),
                np.full(len(dices), power),
                #alpha
                np.full(len(dices), alpha),
            ], dtype='float64')
        else:
            rate = self.dtheta / dt
            thetas = rate * dices + self.theta
            sign = 1 if self.dtheta > 0 else -1

            if craft is not None:
                # degrees above horizontal
                vPitch = -sign * (math.atan2(self.dz, self.length) + alpha * deg2rad)
                
                roll = craft.turnRadiusToBankAngle(alpha, self.radius, height = self.z0 + self.dz / 2, theta = math.atan2(self.dz, self.length)) * deg2rad
                #print('eep', self.z0 + self.dz / 2, math.atan2(self.dz, self.length), roll, roll * rad2deg)
                # FLD = END
                # pitch, roll, yaw
                # Reverse roll because we want CCW here but do CW everywhere else
                # Reverse pitch because down is positive (?)
                # Move 0 to north and rotate the appropriate direction
                headings = sign * (thetas + math.pi / 2)
                #print('head', thetas, headings)
                solarVector = rotate4(0, 0, -1, headings, -roll, vPitch)

                #print('upness', math.atan2(self.dz, self.length)*rad2deg)

                #print(solarVector)

                # N of E to E of N
                solarAzimuth = (math.pi / 2 - np.arctan2(solarVector[1], solarVector[0]) + 2 * math.pi) % (2 * math.pi)
                xyDist = np.sqrt(solarVector[0]**2 + solarVector[1]**2)
                # Angle from vertical
                solarTilt = np.arctan2(-solarVector[2], xyDist)
                solarTilt = -(math.pi/2 - solarTilt)


            return dt, np.array([
                # X Y Z V
                self.cx + self.radius * np.cos(thetas),
                self.cy + self.radius * np.sin(thetas),
                self.z0 + dices * dzdt,
                np.full(len(dices), v),
                # Tilt Azimuth
                #np.full(len(dices), -alpha),
                #(sign * (90 - np.arctan2(np.cos(thetas), -np.sin(thetas)) * rad2deg)) % 360,
                solarTilt * rad2deg, solarAzimuth * rad2deg,
                # Thrust power
                np.full(len(dices), thrust),
                np.full(len(dices), power),
                # Alpha
                np.full(len(dices), alpha),
            ], dtype='float64')

# Can only handle curves, but easier to use (hopefully)
class ExplicitGeneralSegment(GeneralSegment):
    def __init__(self, p0, p1, center, thetaRange):
        from math import pi, sin, cos, atan2, sqrt

        sign = lambda x: x / abs(x) if x != 0 else 0
        signUp = lambda x: 1 if x >= 0 else -1

        if len(p0) != 3 or len(p1) != 3:
            raise TypeError('Please supply a 3-vec')
        
        self.x0, self.y0, self.z0 = self.p0 = [float(i) for i in p0]
        self.x1, self.y1, self.z1 = self.p1 = [float(i) for i in p1]

        dx = self.dx = self.x1 - self.x0
        dy = self.dy = self.y1 - self.y0
        dz = self.dz = self.z1 - self.z0

        # Curve
        #self.groundLength = 2 * pi * self.radius
        #directLength = sqrt(dx ** 2 + dy ** 2)


        # Center coordinates
        self.cx, self.cy = center

        #print('center is', self.cx, self.cy)

        #print('  and back is ', self.y0 - self.cy, self.x0 - self.cx)
        self.theta, self.thetaEnd = thetaRange
        self.dtheta = self.thetaEnd - self.theta
        self.radius = sqrt((self.x0 - self.cx)**2 + (self.y0 - self.cy)**2)

        self.groundLength = abs(self.radius * self.dtheta)
        # good visualization here: https://math.stackexchange.com/questions/2160851/finding-the-length-of-a-helix
        self.length = sqrt(dz ** 2 + self.groundLength ** 2)

        #print('length (ground, total)', self.groundLength, self.length)

        # TODO I think this still works for circular...
        self.ascentAngle = atan2(self.z1 - self.z0, self.groundLength)
        
    
##########################################
##### Helpers for building trajectory ####
##########################################
    
# Angle between two vectors (smallest?)
def angleBetween(a, b):
    return atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1])

def getDir(before, at, after):
    xb, yb, rb, zb = before
    x, y, r, z = at
    xa, ya, ra, za = after
    
    #     lCenter = vector((xb, yb))
    #     hCenter = vector((x, y))
    #     rCenter = vector((xa, ya))
    #delta = rCenter - lCenter
    #hDelta = hCenter - lCenter
    delta = (xa - xb, ya - yb)
    hDelta = (x - xb, y - yb)
    angle = angleBetween(delta, hDelta)
    
    
    return angle > 0

# We need the next two left/right waycircles as well...
def getTangentLineBetween(before, left, right, after):
    cwLeft = getDir(before, left, right)
    cwRight = getDir(left, right, after)
    
    leftFlat = left[0:3]
    rightFlat = right[0:3]
    
    # Just do the left hand side really, next will take care of our right
    if cwLeft and cwRight:
        l1, l2 = getOuterTangent(*leftFlat, *rightFlat)
        if left[2] > right[2]:
            l1, l2 = l2, l1
        return l1
    elif cwLeft and not cwRight:
        l1, l2 = getInnerTangent(*leftFlat, *rightFlat)
        return l2
    elif not cwLeft and cwRight:
        l1, l2 = getInnerTangent(*leftFlat, *rightFlat)
        return l1
    elif not cwLeft and not cwRight:
        l1, l2 = getOuterTangent(*leftFlat, *rightFlat)
        if left[2] < right[2]:
            l1, l2 = l2, l1
        return l1
    else:
        raise Exception("Oops")
    
##########################################
######### The actual trajectory ##########
##########################################

class BaseTrajectory:
    def __init__(self, pieces):
        self.pieces = pieces

    # Create a 3d rendering of the entire path (Sagemath)
    def render(self, **kwargs):
        return sum([p.render(**kwargs) for p in self.pieces])

    def renderColored(self, **kwargs):
        colors = ['red', 'magenta', 'green', 'blue']
        renders = [self.pieces[i].render(**kwargs, color=colors[i % len(colors)]) for i in range(len(self.pieces))]
        return sum(renders)

    # 2d renderings
    def renderTop(self, **kwargs):
        return sum([p.renderTop(**kwargs) for p in self.pieces])

    def renderSide(self, **kwargs):
        return sum([p.renderSide(**kwargs) for p in self.pieces])
    
    # Length of our entire path
    def length(self):
        return sum([p.length for p in self.pieces])


class WaycircleTrajectory(BaseTrajectory):
    def __init__(self, wayCircles):
        if len(wayCircles) < 2:
            raise IndexError('You should probably have at least 2 wayCircles')
            
        # Perturb any equal adjacent radii just a bit
        # TODO fix our tangent generation code to handle same-radius circles
        for j in range(2):
            for i in range(-1, len(wayCircles) - 1):
                if wayCircles[i][2] == wayCircles[i + 1][2]:
                    # Hopefully fine...
                    wayCircles[i][2] += 0.01
                    #raise TypeError('Sorry, adjacent radii must be distinct!')
        
        pieces = self.buildTrajectory(wayCircles)
        super().__init__(pieces)
        
    def buildTrajectory(self, wayCircles):
        pieces = []
        for i in range(len(wayCircles)):
            nextBefore = wayCircles[i - 2]
            before = wayCircles[i - 1]
            at = wayCircles[i]
            after = wayCircles[(i + 1) % len(wayCircles)]
            nextAfter = wayCircles[(i + 2) % len(wayCircles)]

            # The lines before/after this waycircle
            left = getTangentLineBetween(nextBefore, before, at, after)
            right = getTangentLineBetween(before, at, after, nextAfter)
            
            x, y, r, z = at

            # The points at the ends of those lines
            pt0 = vector(left[1])
            pt1 = vector(right[0])

            center = vector((x, y))

            # Angles from center to tangent points
            dTheta = angleBetween(pt0 - center, pt1 - center)
            theta0 = atan2(pt0[1] - y, pt0[0] - x)
            theta1 = theta0 + dTheta
            
            # Have to correct some circles that get spun in the wrong direction
            # TODO actually fix this
            cw = getDir(before, at, after)
            
            if cw and dTheta > 0:
                theta1 -= 2 * pi
            if not cw and dTheta < 0:
                theta1 += 2 * pi

            # Add our pieces to the list
            zNext = after[3]
            pieces.append(LineSegment(left[0][0], left[0][1], z, left[1][0], left[1][1], zNext))
            pieces.append(ArcSegment(x, y, zNext, r, theta0, theta1 - theta0))
        
        return pieces
        
# Waycircle trajectory, but hopefully more amenable to stochastic methods and a 24-hour period
class ClampedVectorTrajectory(BaseTrajectory):
    def __init__(self, vec, craft, startPosition = (0, 0, 2000), time_limit = 86400):
        from math import sqrt, tan
        # Something to loop easier: https://stackoverflow.com/a/5389547
        def grouped(iterable, n):
            return zip(*[iter(iterable)]*n)
        # Current plan: vector of offsets (and radii, alphas) to generate waycircles
        # Initial and final points are fake circles with a small radius and aren't generated (just so we can use the waycircle code)
        # We generate segments until we hit 24 hours of flight time
        # Pros: 
        #   24 hours guaranteed (probably)
        #   no need for loopiness
        # Cons: 
        #   we have to know about the aircraft
        #   Earlier points impact all later points
        #   Basically mandates constraint violation
        #   Can have non-coding inputs (when exceeding 24 hours)

        # Vector representation: Array<Item>
        # Item [ deltaX, deltaY, theta, radius, alphaLine, alphaCurve ]
        
        startX, startY, startZ = startPosition

        time_at = 0
        alphas = []
        thetas = []

        # Build up our waycircles first
        # X Y R Z
        wayCircles = [
            (startX-2, startY-2, 1, startZ),
            (startX, startY, 1, startZ)
        ]

        for deltaX, deltaY, theta, radius, alphaLine, alphaCurve in grouped(vec[:-6], 6):
            lx, ly, lr, lz = wayCircles[-1]
            dist = sqrt(deltaX ** 2 + deltaY ** 2)
            deltaZ = dist * tan(theta)

            #print('Moving %.2f and had %.2f and %.2f' % (dist, radius, lr))

            wayCircles.append((lx + deltaX, ly + deltaY, radius, lz + deltaZ))

            alphas.append(alphaLine)
            alphas.append(alphaCurve)

        lx, ly, lr, lz = wayCircles[-1]
        deltaX, deltaY, theta, radius, alphaLine, alphaCurve = vec[-6:]
        lx += deltaX
        ly += deltaY
        wayCircles.append((lx, ly, 1, lz))
        wayCircles.append((lx+2, ly+2, 1, lz))

        for i in range(len(wayCircles) - 1):
            l, r = wayCircles[i:i+2]
            lx, ly, lr, lz = l
            rx, ry, rr, rz = r

            dist = sqrt((lx - rx)**2 + (ly - ry)**2)
            rads = lr + rr

            # Adjust the radii such that there is no overlap
            if rads > dist:
                #print('oops', i, l, r, dist, rads)
                factor = rads / dist
                #print('  adjusting radius by ', factor)
                wayCircles[i] = (lx, ly, lr / factor - 1, lz)
                wayCircles[i+1] = (rx, ry, rr / factor - 1, rz)


        # Now convert to a trajectory like before, but DON'T close the loop
        pieces = []
        for i in range(2, len(wayCircles) - 2):
            nextBefore = wayCircles[i - 2]
            before = wayCircles[i - 1]
            at = wayCircles[i]
            after = wayCircles[(i + 1) % len(wayCircles)]
            nextAfter = wayCircles[(i + 2) % len(wayCircles)]

            # The lines before/after this waycircle
            left = getTangentLineBetween(nextBefore, before, at, after)
            right = getTangentLineBetween(before, at, after, nextAfter)
            
            x, y, r, z = at

            # The points at the ends of those lines
            pt0 = vector(left[1])
            pt1 = vector(right[0])

            center = vector((x, y))

            # Angles from center to tangent points
            dTheta = angleBetween(pt0 - center, pt1 - center)
            theta0 = atan2(pt0[1] - y, pt0[0] - x)
            theta1 = theta0 + dTheta
            
            # Have to correct some circles that get spun in the wrong direction
            # TODO actually fix this
            cw = getDir(before, at, after)
            
            if cw and dTheta > 0:
                theta1 -= 2 * pi
            if not cw and dTheta < 0:
                theta1 += 2 * pi

            # Add our pieces to the list
            zNext = after[3]
            pieces.append(LineSegment(left[0][0], left[0][1], z, left[1][0], left[1][1], zNext))
            pieces.append(ArcSegment(x, y, zNext, r, theta0, theta1 - theta0))

        # Now clamp the trajectory
        # TODO technically it would be more efficient to do above...
        time_at = 0
        alpha_at = 0
        for piece in pieces:
            alpha = alphas[alpha_at]
            

            p = 0
            
            while p == 0:
                if isinstance(piece, LineSegment):
                    # Take mean altitude
                    h = (piece.z0 + piece.z1) / 2
                    try:
                        v, t, p = craft.fastStraightVelocityThrustPower(piece.theta, alpha, h)
                    except:
                        print(piece.theta, alpha, h)
                        raise
                elif isinstance(piece, ArcSegment):
                    #print(piece.r, alpha)
                    v, t, p = craft.fastTurningVelocityThrustPower(piece.r, alpha, piece.z)
                else:
                    #print(piece)
                    raise TypeError('Whats that piece?')
                    
                if p == 0:
                    #print('it all went wrong: ', piece.theta, alpha, h)
                    # If we are not using any power (aka "falling"), increase our angle of attack until we are not
                    alphas[alpha_at] += 0.25
                    alpha = alphas[alpha_at]
                    #print('  beep', alpha, v, p)

            alpha_at += 1
            dt = piece.length / v

            time_at += dt
            
            if time_at > time_limit:
                #print('times up!')
                break

            

        #print('total time is', time_at)
        
        super().__init__(pieces)
        self.alphas = alphas
        self.time = time_at
            

class CircleTrajectory(BaseTrajectory):
    '''Simple circular trajectory'''
    def __init__(self, center, radius, phase = 0, reverse = False):
        super().__init__([
            ArcSegment(*center, radius, phase, (-1 if reverse else 1) * 2 * pi)
        ])

def ArcSegmentFromCenterAndPoints(center, left, right, flip = False):
    from math import pi, atan2
    # Assume that left and right are equidistant from center
    radius = sqrt((center[0] - left[0])**2 + (center[1] - left[1])**2)
    leftDelta = (
        left[0] - center[0],
        left[1] - center[1]
    )
    rightDelta = (
        right[0] - center[0],
        right[1] - center[1]
    )
    theta = atan2(leftDelta[1], leftDelta[0])
    # Take small angle, and invert to get outside angle
    dTheta =  angleBetween(leftDelta, rightDelta)
    if dTheta < 0: dTheta += 2 * pi
    if flip: dTheta -= 2 * pi
    return ArcSegment(*center, radius, theta, dTheta)

class BowtieTrajectory(BaseTrajectory):
    '''Bowtie, or figure 8, trajectory'''
    def __init__(self, center, lobeAngle = 0, lobeRadius = 50, lobeCenterDistance = 100):
        from math import sin, cos

        cx, cy, cz = center
        
        # Just two lobes for now
        c1 = (
            cx + lobeCenterDistance * cos(lobeAngle),
            cy + lobeCenterDistance * sin(lobeAngle),
            cz
        )

        c2 = (
            cx - lobeCenterDistance * cos(lobeAngle),
            cy - lobeCenterDistance * sin(lobeAngle),
            cz
        )

        left, right = getInnerTangent(c1[0], c1[1], lobeRadius, c2[0], c2[1], lobeRadius)
        super().__init__([
            LineSegment(*left[1], cz, *left[0], cz),
            ArcSegmentFromCenterAndPoints(c1, left[0], right[0], True),
            LineSegment(*right[0], cz, *right[1], cz),
            ArcSegmentFromCenterAndPoints(c2, right[1], left[1]),
        ])
    


class SimpleLadderTrajectory(BaseTrajectory):
    ''' Bowtie, but climbing and then descending '''
    def __init__(self, center, lobeAngle = 0, lobeRadius = 50, lobeCenterDistance = 100,
                 stepHeight = 5, nSteps = 2, nStepsDown = None):
        

        if nStepsDown is None:
            nStepsDown = nSteps


        cx, cy, cz = center

        c1 = (
            cx + lobeCenterDistance * cos(lobeAngle),
            cy + lobeCenterDistance * sin(lobeAngle),
            cz
        )

        c2 = (
            cx - lobeCenterDistance * cos(lobeAngle),
            cy - lobeCenterDistance * sin(lobeAngle),
            cz
        )

        left, right = getInnerTangent(c1[0], c1[1], lobeRadius, c2[0], c2[1], lobeRadius)
    
        pieces = []

        ladderHeight = stepHeight * max(nSteps, nStepsDown)

        stepUpHeight = ladderHeight / nSteps
        stepDownHeight = ladderHeight / nStepsDown

        for i in range(nSteps):
            # Height at bottom, mid/other side, and step above for this step, respectively
            zl = cz + (2 * i + 0) * stepUpHeight
            zm = cz + (2 * i + 1) * stepUpHeight
            zr = cz + (2 * i + 2) * stepUpHeight
            # Centers for the two sides of this step
            cl = (c1[0], c1[1], zm)
            cr = (c2[0], c2[1], zr)

            pieces.extend([
                LineSegment(*left[1], zl, *left[0], zm),
                ArcSegmentFromCenterAndPoints(cl, left[0], right[0], True),
                LineSegment(*right[0], zm, *right[1], zr),
                ArcSegmentFromCenterAndPoints(cr, right[1], left[1]),
            ])

        for i in range(nStepsDown-1, -1, -1):
            zl = cz + (2 * i + 2) * stepDownHeight
            zm = cz + (2 * i + 1) * stepDownHeight
            zr = cz + (2 * i + 0) * stepDownHeight
            cl = (c1[0], c1[1], zm)
            cr = (c2[0], c2[1], zr)

            pieces.extend([
                LineSegment(*left[1], zl, *left[0], zm),
                ArcSegmentFromCenterAndPoints(cl, left[0], right[0], True),
                LineSegment(*right[0], zm, *right[1], zr),
                ArcSegmentFromCenterAndPoints(cr, right[1], left[1]),
            ])

        super().__init__(pieces)

    def render(self, cutoff = -1, **kwargs):
        return sum([p.render(**kwargs) for p in self.pieces[:cutoff]])

    def renderSideFancy(self, cutoff, **kwargs):
        ret = [piece.renderSide(**kwargs) for piece in self.pieces[:cutoff]]

        ret.append(self.pieces[cutoff].renderSide(linestyle='--', **kwargs))

        # Eh
        ret.append(self.pieces[-1].renderSide(**kwargs))

        return sum(ret)


class DTrajectory(BaseTrajectory):
    # Like the letter "D", but with rounded corners
    def __init__(self, center, radius, cornerRadius, angle = 0, reverse = False):
        from math import sin, cos, atan2, pi, asin

        cx, cy, cz = center

        # Radius to centers of the rounded corners
        innerRadius = radius - cornerRadius
        # Angle between edge and line to centers of the rounded corners
        innerTheta = asin(cornerRadius / innerRadius)

        leftCenter = (
            cx + innerRadius * cos(angle + pi - innerTheta),
            cy + innerRadius * sin(angle + pi - innerTheta),
            cz
        )

        rightCenter = (
            cx + innerRadius * cos(angle + innerTheta),
            cy + innerRadius * sin(angle + innerTheta),
            cz
        )

        edgeRadius = sqrt(innerRadius**2 - cornerRadius**2)
        leftEdgePoint = (
            cx + edgeRadius * cos(angle + pi),
            cy + edgeRadius * sin(angle + pi)
        )

        rightEdgePoint = (
            cx + edgeRadius * cos(angle),
            cy + edgeRadius * sin(angle)
        )

        leftOuterPoint = (
            cx + radius * cos(angle + pi - innerTheta),
            cy + radius * sin(angle + pi - innerTheta)
        )

        rightOuterPoint = (
            cx + radius * cos(angle + innerTheta),
            cy + radius * sin(angle + innerTheta)
        )

        print(innerTheta)

        if reverse:
            # CW
            super().__init__([
                LineSegment(*leftEdgePoint, cz, *rightEdgePoint, cz),
                ArcSegmentFromCenterAndPoints(rightCenter, rightEdgePoint, rightOuterPoint),
                ArcSegmentFromCenterAndPoints(center, rightOuterPoint, leftOuterPoint),
                ArcSegmentFromCenterAndPoints(leftCenter, leftOuterPoint, leftEdgePoint)
            ])
        else:
            # CCW
            super().__init__([
                LineSegment(*rightEdgePoint, cz, *leftEdgePoint, cz),
                ArcSegmentFromCenterAndPoints(leftCenter, leftEdgePoint, leftOuterPoint, True),
                ArcSegmentFromCenterAndPoints(center, leftOuterPoint, rightOuterPoint, True),
                ArcSegmentFromCenterAndPoints(rightCenter, rightOuterPoint, rightEdgePoint, True),
            ])


# Freestyling modified circular trajectory (not smooth!)
class DeltaThetaZScheduleTrajectory(BaseTrajectory):
    def __init__(self, craft, controlPoints, initialTheta = 0, initialRadius = 1000, initialHeight = 1000, zSchedule=[6*3600]*4, gain = 0):
        from math import sin, cos, sqrt, tan, atan2
        pieces = []
        times = []

        lastTheta, lastRadius = initialTheta, initialRadius
        time_at = 0
        height_at = initialHeight

        ascend, sustain, descend, rest = zSchedule

        period_changes = [ascend, ascend + sustain, ascend + sustain + descend]
        slopes = [gain / ascend, 0, -gain / descend, 0]
        thetas = [atan2(el, 1) for el in slopes]
        time_period = 0

        #print('delta', period_changes)

        # This is updated by fixed-point guesswork later
        # storing it outside allows fast convergence because we usually will have a very good guess to start
        thetaGuess = 0.05

        for deltaTheta, nextRadius, nextCurvature in controlPoints:
            nextTheta = lastTheta + deltaTheta
            #print('at', nextTheta)

            # TODO Z scheduling

            lx = lastRadius * cos(lastTheta)
            ly = lastRadius * sin(lastTheta)

            rx = nextRadius * cos(nextTheta)
            ry = nextRadius * sin(nextTheta)

            

            # Bleh
            # curvature 0 should make a large radius ( |radius| >> 0 )
            # Curvature 1 should make an outside semicircle (radius = minRad)
            # Curvature -1 should make an inside semicircle (radius = -minRad)
            directLength = sqrt((rx-lx)**2 + (ry-ly)**2)
            minRad = directLength / 2
            radius = 1.01 * minRad / nextCurvature if nextCurvature != 0 else minRad * 1e9
            #print('!!!', minRad, radius)

            center, thetaInfo = pointToPointArc((lx, ly, 0), (rx, ry, 0), radius)
            theta, dtheta = thetaInfo
            
            lz = height_at
            slope = slopes[time_period]
            theta = thetas[time_period]
            
            # We might be able to solve for theta, but fixed point converges really quick
            
            for i in range(3):
                v, t, p = craft.fastGeneralVelocityThrustPower(alpha = 5, theta=thetaGuess, radius=radius, height=height_at)
                groundLength = abs(dtheta) * radius
                dt = groundLength / v
                # This will not be exact, as the real length will be greater slightly due to increasing altitude, but close enough
                dz = dt * slope
                
                rz = height_at + dz

                left = (lx, ly, lz)
                right = (rx, ry, rz)

                seg = GeneralSegment(left, right, radius)

                #print('  %s->%s (%.2f, %.2f, %s)' % (thetaGuess, seg.ascentAngle, v, dz, abs(thetaGuess - seg.ascentAngle)))
                if abs(thetaGuess - seg.ascentAngle) < 0.001:
                    break

                thetaGuess = seg.ascentAngle

            height_at += dz

            if height_at < 0:
                import numpy as np
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()
                ax.plot(times, np.array([piece.z0 for piece in pieces]))
                for x in period_changes:
                    ax.axvline(x)
                print('zSchedule', sum(zSchedule), zSchedule, slopes, gain)
                print(groundLength, dt, rz, left, right, height_at, dz)
                raise TypeError('We went underground!')

            #print('from %s to %s' % (left, right))

            if abs(nextCurvature) > 1:
                raise TypeError('Curvature must be in [-1, 1] (%.2f)' % nextCurvature)

            

            # TODO To make this fit with reality, we'd want to add in some sort of roundover like curve at sharp corners
            # OR, we could just make such things have a penalty and let nature take its course

            
            pieces.append(seg)

            #print('guess %s vs actual %s' % (0.01, seg.ascentAngle))

            time_at += seg.length / v
            times.append(time_at)
            if time_at > 86400:
                # Finished the day early
                #print('the end')
                break

            if time_period < 3 and time_at > period_changes[time_period]:
                #print('next at', time_at)
                time_period += 1

            lastTheta, lastRadius = nextTheta, nextRadius

        super().__init__(pieces)


class ImperativeTrajectory(BaseTrajectory):
    def __init__(self, craft, startPosition, startHeading, commands, commandDuration = 20):
        # Pros: direct(ish) control of aircraft attitude/speed and total duration
        # Cons: high impact of earlier segments, must be manually constrained
        pos_at, theta = startPosition, startHeading
        pieces = []

        for alpha, bank, dz in commands:
            piece, pos_at, theta = self.makeSegment(craft, pos_at, theta, alpha, bank, dz, commandDuration)
            pieces.append(piece)

        self.end_position = pos_at
        self.end_heading = theta
        
        super().__init__(pieces)

    def makeSegment(self, craft, pos_at, initial_heading, alpha, bank, dz, dt = 20):
        from thesis.Trajectory import ExplicitGeneralSegment
        from math import atan2, sin, cos

        def sign(v):
            return v / abs(v) if v != 0 else 0
        
        # Sanity
        if bank == 0:
            bank = 0.001
        
        lx, ly, lz = pos_at

        if dz == 0:
            theta = 0
            radius = craft.bankAngleToTurnRadius(alpha, bank, theta = theta, height=lz)
            v, t, p = craft.fastGeneralVelocityThrustPower(alpha = alpha, theta = theta, radius = radius, height=lz)
            
        else:
            theta = 0.1 * sign(dz)
            # A quick fixed point to find our theta s.t. we ascend/descend the proper amount
            for i in range(5):
                radius = craft.bankAngleToTurnRadius(alpha, bank, theta = theta, height=lz)
                v, t, p = craft.fastGeneralVelocityThrustPower(alpha = alpha, theta = theta, radius = radius, height=lz)
                
                dzCurrent = v * dt * sin(theta)
                
                actualTime = dt * (dzCurrent / dz)
                if abs(dz - dzCurrent) < 0.0001:
                    break
                theta = theta * (dz / dzCurrent)
            
        #print('vel is', v, 'radius is', radius, 'theta is', theta)
            
        # Compute our center/ending points
        quarterCircle = pi / 2
        toCenter = initial_heading - sign(bank) * quarterCircle
        #print('>', toCenter)
        cx = lx + abs(radius) * cos(toCenter)
        cy = ly + abs(radius) * sin(toCenter)
        
        groundLength = v * dt * cos(theta)
        trajDeltaTheta = -groundLength / radius
        trajThetaStart = atan2(ly - cy, lx - cx)
        trajThetaEnd = trajThetaStart + trajDeltaTheta
        
        rx = cx + abs(radius) * cos(trajThetaEnd)
        ry = cy + abs(radius) * sin(trajThetaEnd)
        rz = lz + v * dt * sin(theta)
        
        pos_end = (rx, ry, rz)
        
        #print('ground', groundLength)
        #print('%.2f %.2f %.2f' % (cx, cy, radius))

        center = (cx, cy)
        thetaRange = (trajThetaStart, trajThetaEnd)
        return ExplicitGeneralSegment(pos_at, pos_end, center, thetaRange), pos_end, initial_heading + trajDeltaTheta