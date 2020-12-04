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
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        
        self.x1 = x1
        self.y1 = y1
        self.z1 = z1
        
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
#         v = n(craft.straightVelocity(θ = self.theta, α=alpha, a=0))
#         t = n(craft.straightThrust(θ = self.theta, α=alpha, a=0))
#         p = n(craft.straightPower(θ = self.theta, α=alpha, a=0))
        
#         return v, t, p

        return craft.fastStraightVelocityThrustPower(θ = self.theta, α=alpha, a=0)
        
class ArcSegment:
    def __init__(self, x, y, z, r, theta, dtheta):
        self.x = x
        self.y = y
        self.z = z
        self.r = r
        self.theta = theta.n()
        self.dtheta = dtheta.n()
        
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
        ])
    
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
#         v = n(craft.turningVelocity(r = self.r, α=alpha))
#         t = n(craft.turningThrust(r = self.r, α=alpha))
#         p = n(craft.turningPower(r = self.r, α=alpha))
        
#         return v, t, p
        return craft.fastTurningVelocityThrustPower(self.r, alpha)
    
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
    
class Trajectory:
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
        
        self.buildTrajectory(wayCircles)
        
    def buildTrajectory(self, wayCircles):
        self.pieces = []
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
            self.pieces.append(LineSegment(left[0][0], left[0][1], z, left[1][0], left[1][1], zNext))
            self.pieces.append(ArcSegment(x, y, zNext, r, theta0, theta1 - theta0))
        
    # Create a 3d rendering of the entire path (Sagemath)
    def render(self):
        return sum([p.render() for p in self.pieces])
    
    # Length of our entire path
    def length(self):
        return sum([p.length for p in self.pieces])