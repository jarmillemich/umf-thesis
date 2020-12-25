# a class for representing our "Waycircle" trajectories
# along with supporting utilities (including sage utils)
from sage.all import (
    vector, sqrt, atan2, line, arc,
    cos, sin, n, minimize, find_root,
    show, plot, RealSet, var, pi
)
#from math import sqrt, atan2


# Get tangent points for circles
# http://www.ambrsoft.com/TrigoCalc/Circles2/Circles2Tangent_.htm ...
def getTangentWithPoint(x0, y0, r0, x1, y1, r1, xp, yp):
    # In the symbology of that page
    a, b, c, d = x0, y0, x1, y1
    
    # These are for the first circle
    denom0 = (xp - a)**2 + (yp - b)**2
    sqrtTerm0 = sqrt(denom0 - r0**2)
    
    
    xt1 = (r0**2 * (xp - a) + r0 * (yp - b) * sqrtTerm0) / denom0 + a
    xt2 = (r0**2 * (xp - a) - r0 * (yp - b) * sqrtTerm0) / denom0 + a
    
    yt1 = (r0**2 * (yp - b) - r0 * (xp - a) * sqrtTerm0) / denom0 + b
    yt2 = (r0**2 * (yp - b) + r0 * (xp - a) * sqrtTerm0) / denom0 + b
    
    # These are for the second circle
    denom1 = (xp - c)**2 + (yp - d)**2
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

        return craft.fastStraightVelocityThrustPower(θ = self.theta, α=alpha, a=0, h=h)

    ## Experimental
    def toPoses(self, times, t0, v, alpha):
        import pandas as pd
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
        #print(dt, len(slices), slices)
        ret = pd.DataFrame({
            'x': [self.x0 + dx * (t - t0).total_seconds() / dt for t in slices],
            'y': [self.y0 + dy * (t - t0).total_seconds() / dt for t in slices],
            'z': [self.z0 + dz * (t - t0).total_seconds() / dt for t in slices],
            'v': [v for t in slices],
            'tilt': [tilt for t in slices],
            'azimuth': [azimuth % 360 for t in slices]

        }, index=slices, dtype=float)

        # Extra millisecond to avoid overlap
        return endTime + pd.offsets.Milli(1), ret
        
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
        xdata = [x + r * cos(s1 + t * dt) for t in range(n)]
        ydata = [y + r * sin(s1 + t * dt) for t in range(n)]

        return line([
            (xdata[i], ydata[i], z)
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
    def toPoses(self, times, t0, v, alpha):
        import pandas as pd
        import math
        dt = self.length / v
        rate = self.dtheta / dt

        # Apparently can't have fractional times, so do milliseconds
        endTime = t0 + pd.offsets.Milli(int(dt * 1e3))

        # We are assuming that the solar panels are in-line with the wings
        # North is +Y
        # Azimuth is degrees east of north
        rad2deg = 180 / math.pi
        azimuth = 0
        tilt = 0

        slices = times[t0:endTime]
        thetas = [rate * (t - t0).total_seconds() + self.theta for t in slices]

        sign = 1 if self.dtheta > 0 else -1


        ret = pd.DataFrame({
            'x': [self.x + self.r * math.cos(theta) for theta in thetas],
            'y': [self.y + self.r * math.sin(theta) for theta in thetas],
            'z': [self.z for t in slices],
            'v': [v for t in slices],
            'tilt': [-alpha for t in slices],
            # TODO this needs to be totally reworked to take into account our roll AND alpha
            'azimuth': [(sign * (90 - math.atan2(math.cos(theta), -math.sin(theta)) * rad2deg)) % 360 for theta in thetas]
        }, index=slices, dtype=float)

        # Extra millisecond to avoid overlap
        return endTime + pd.offsets.Milli(1), ret
    
##########################################
##### Helpers for building trajectory ####
##########################################
    
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
    
    # Length of our entire path
    def length(self):
        return sum([p.length for p in self.pieces])

    def toPoses(self):
        t0 = 0


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
                 stepHeight = 5, nSteps = 2):
        
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

        for i in range(nSteps):
            zl = cz + (2 * i + 0) * stepHeight
            zm = cz + (2 * i + 1) * stepHeight
            zr = cz + (2 * i + 2) * stepHeight
            cl = (c1[0], c1[1], zm)
            cr = (c2[0], c2[1], zr)

            pieces.extend([
                LineSegment(*left[1], zl, *left[0], zm),
                ArcSegmentFromCenterAndPoints(cl, left[0], right[0], True),
                LineSegment(*right[0], zm, *right[1], zr),
                ArcSegmentFromCenterAndPoints(cr, right[1], left[1]),
            ])

        for i in range(nSteps-1, -1, -1):
            zl = cz + (2 * i + 2) * stepHeight
            zm = cz + (2 * i + 1) * stepHeight
            zr = cz + (2 * i + 0) * stepHeight
            cl = (c1[0], c1[1], zm)
            cr = (c2[0], c2[1], zr)

            pieces.extend([
                LineSegment(*left[1], zl, *left[0], zm),
                ArcSegmentFromCenterAndPoints(cl, left[0], right[0], True),
                LineSegment(*right[0], zm, *right[1], zr),
                ArcSegmentFromCenterAndPoints(cr, right[1], left[1]),
            ])

        super().__init__(pieces)

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
