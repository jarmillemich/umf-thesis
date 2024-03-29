# Given some waypoints (with headings), make a bunch of arcs such that:
# Between each waypoint are two circlular arcs, tangent to each other and tangent
#   to each heading at the waypoints and oriented such that there is a smooth flight path
# If desiredDuration and craft are specified, will scale about center until the desired time is reached (hopefully)

from thesis.Trajectory import BaseTrajectory, ExplicitGeneralSegment

def rescale(points, center, scale):
  """Scale X/Y coordinates around the given center"""
  cx, cy = center

  def rescaleSingle(pt):
    if len(pt) == 6:
      x, y, z, h, a1, a2 = pt
    elif len(pt) == 4:
      x, y, z, h = pt
      a1, a2 = 5, 5
    else:
      raise TypeError('bad rescale point')
    x -= cx
    y -= cy
    x *= scale
    y *= scale
    x += cx
    y += cy
    return (x, y, z, h, a1, a2)

  return [rescaleSingle(pt) for pt in points]

from math import sin, cos, sqrt, pi
from thesis.optimize.BaseOptimizer import Vector
def subSolve(leftInfo, rightInfo, rightFirst):
  """Solve the bi-arc like curves for the given info"""
  # Which way from the two points the centers are (+/- radian direction)
  rightSide = 1 if rightFirst else -1
  leftSide = -1 if rightFirst else 1
  
  quarter = pi / 2
  
  left = Vector(leftInfo[0:2])
  theta1 = leftInfo[3]
  leftDir = Vector([cos(theta1), sin(theta1)])
  
  right = Vector(rightInfo[0:2])
  theta2 = rightInfo[3]
  rightDir = Vector([cos(theta2), sin(theta2)])
  
  # If our two thetas are nearly parallel, add in a small
  # correction factor to avoid a degenerate case
  corr = 0
  if abs(theta1 - theta2) < 0.01: corr = 0.02
    
  theta1Prime = theta1 + corr + leftSide * quarter
  theta2Prime = theta2 + rightSide * quarter
  
  u = cos(theta1Prime) - cos(theta2Prime)
  v = left[0] - right[0]
  w = sin(theta1Prime) - sin(theta2Prime)
  x = left[1] - right[1]
  
  a = u**2 + w**2 - 4
  b = 2 * (u * v + w * x)
  c = v**2 + x**2
  
  r1 = (-b + sqrt(b**2 - 4 * a * c)) / (2 * a)
  r2 = (-b - sqrt(b**2 - 4 * a * c)) / (2 * a)
  
  if r1 > 0 and r2 > 0:
    raise LogicError('Should not occur')
    
  r = r1 if r1 > 0 else r2
  
  c1 = left + r * leftDir.rotate(leftSide * quarter + corr)
  c2 = right + r * rightDir.rotate(rightSide * quarter)
  
  # The tangent point of the two circles
  middle = (c1 + c2) / 2
  
  # Determines which way around the circles we go (our theta ranges)
  toLeft = left - c1
  toLeftMiddle = middle - c1
  toRight = right - c2
  toRightMiddle = middle - c2
  
  
  leftThetaRange = [toLeft.angle(), toLeftMiddle.angle()]
  # XXX This is moderately opaque...
  #if leftSide < 0: leftThetaRange = leftThetaRange[::-1]
  #if rightFirst and leftThetaRange[1] > leftThetaRange[0]: leftThetaRange[0] += 2 * pi
  #if not rightFirst and leftThetaRange[1] < leftThetaRange[0]: leftThetaRange[1] += 2 * pi
    
  rightThetaRange = [toRightMiddle.angle(), toRight.angle()]
  #if rightSide > 0: rightThetaRange = rightThetaRange[::-1]
  #if rightFirst: rightThetaRange[0] += 2 * pi
  #if rightSide > 0 and rightThetaRange[1] < rightThetaRange[0]: rightThetaRange[1] += 2 * pi
  #if not rightFirst and leftThetaRange[1] > rightThetaRange[0]: rightThetaRange[0] += 2 * pi
  
  # This is stupid, make CW
  #print(leftThetaRange, rightThetaRange)
  
  leftDir = leftThetaRange[1] - leftThetaRange[0]
  rightDir = rightThetaRange[1] - rightThetaRange[0]
  
  # Should be going CW first but not
  if rightFirst: # CW and then CCW
    if leftDir > 0: # CCW but should be CW
      leftThetaRange[1] -= 2 * pi
    if rightDir < 0: # CW but should be CCW
      rightThetaRange[1] += 2 * pi
  else: # CCW and then cW
    if leftDir < 0: # CW but should be CCW
      leftThetaRange[1] += 2 * pi
    if rightDir > 0: # CCW but should be CW
      rightThetaRange[1] -= 2 * pi
  
  #if leftThetaRange[1] < leftSide * leftThetaRange[0]: leftThetaRange[1] += 2 * pi
  #if rightThetaRange[1] < rightSide * rightThetaRange[0]: rightThetaRange[1] += 2 * pi
    
  # Eh
  #leftThetaDist = (leftThetaRange[1] - leftThetaRange[0] + 2 * pi) % (2 * pi)
  #rightThetaDist = (rightThetaRange[1] - rightThetaRange[0] + 2 * pi) % (2 * pi)
  
  leftThetaDist = abs(leftThetaRange[1] - leftThetaRange[0])
  rightThetaDist = abs(rightThetaRange[1] - rightThetaRange[0])
  
  leftWeight = leftThetaDist / (leftThetaDist + rightThetaDist)
  rightWeight = rightThetaDist / (leftThetaDist + rightThetaDist)
  
  leftHeight = leftInfo[2]
  rightHeight = rightInfo[2]
  dz = rightHeight - leftHeight
  midHeight = leftHeight + dz * leftWeight
  #print(leftThetaDist, rightThetaDist, leftWeight, rightWeight, midHeight)
  
  #print(toLeftMiddle, leftThetaRange)
  
  leftSegment = ExplicitGeneralSegment(leftInfo[0:3], [*middle, midHeight], c1, leftThetaRange)
  rightSegment = ExplicitGeneralSegment([*middle, midHeight], rightInfo[0:3], c2, rightThetaRange)
  
  return leftSegment, rightSegment

def solve(left, right, minimumRadius = None):
  """Solve both left-first and right-first arcs and return the shorter greater than the minimumRadius"""
  a = subSolve(left, right, True)
  b = subSolve(left, right, False)
  
  if minimumRadius is not None:
    if a[0].radius < minimumRadius:
      if b[0].radius < minimumRadius:
        raise TypeError('Too close! %.2f,%.2f<%.2f' % (a[0].radius, b[0].radius, minimumRadius))
      return b

  # Return the shorter of the two paths
  if a[0].length + a[1].length < b[0].length + b[1].length:
    return a
  else:
    return b

class SplineyTrajectory(BaseTrajectory):
  """
  Construct our "Spliney" trajectory.

  The trajectory is made up of a series of waypoints, which describe
  a position and heading. Between each waypoint two circular arcs of
  equal radius are created, such that the circles are tangent to the
  headings at their positions, and each other. The parameters of the
  circles are chosen such that the path is continuous, and such that
  the shortest path of the two valid solutions is returned. Segments
  are thus dependent only on the waypoints immediately adjacent.
  """
  def __init__(
    self,
    waypoints,
    center = (0, 0),
    desiredDuration = None,
    craft = None,
    minimumRadius = 50,
    initialHeight = 1000,
    fixedFirst = False,
    zSchedule = None
  ):

    currentScale = 1

    doFixedPoint = desiredDuration is not None and craft is not None
    self.alphas = []

    if zSchedule is not None:
      waypoints = self.doZSchedule(waypoints, zSchedule, craft)
      #print(waypoints)

    # Run a number of iterations of fixed point (should converge)
    # Or, just one if not doing fixed point
    for it in range(10 if doFixedPoint else 1):
      mappedPoints = rescale(waypoints, center, currentScale)
      if fixedFirst:
        mappedPoints[0] = waypoints[0]

      segments = []
      time = 0
      lastZ = initialHeight
      for i in range(len(waypoints) - 1):
        # Generate our segments
        x0, y0, z0, h0, a1, a2 = mappedPoints[i]
        segs = solve(mappedPoints[i], mappedPoints[i + 1], minimumRadius = minimumRadius)
        segments.extend(segs)

        
        if it == 0: self.alphas.extend([a1, a2])

        if doFixedPoint:
          # Calculate our path time
          vl = segs[0].velocityThrustPower(craft, a1)[0]
          vr = segs[1].velocityThrustPower(craft, a2)[0]
        
          time += segs[0].length / vl + segs[1].length / vr
      
      if doFixedPoint:
        # Break early if we are close
        if abs(desiredDuration - time) < 0.001:
          break

        # Update our scale towards the desired value
        currentScale *= desiredDuration / time
        self.scale = currentScale

    super().__init__(segments)

  def doZSchedule(self, waypoints, zSchedule, craft):
    # Gather up the relative times of all of our segments
    import numpy as np
    gain, *schedule = zSchedule
    total_time = 0
    times = []
    for i in range(len(waypoints) - 1):
      x0, y0, h0, a1, a2 = waypoints[i]
      x1, y1, h1, _, _ = waypoints[i + 1]
      segs = solve([x0, y0, 1000, h0, a1, a2], [x1, y1, 1000, h1])
      times.append(total_time)
      total_time += segs[0].length / segs[0].velocityThrustPower(craft, a1)[0]
      total_time += segs[1].length / segs[1].velocityThrustPower(craft, a2)[0]

    duration = 24 * 60 * 60
    times = np.array(times) / total_time

    # Normalize our z Schedule
    schedule = np.array(schedule)
    schedule /= schedule.sum()

    rest, ascend, sustain, descend, moreRest = schedule
    ascendStart = rest
    sustainStart = ascendStart + ascend
    descendStart = sustainStart + sustain
    restStart = descendStart + descend

    # Start slicing and dicing
    out = []
    z0 = 1000 # TODO hardcoding
    for i in range(len(waypoints)):
      x, y, h, a1, a2 = waypoints[i]

      z = z0
      
      if i == len(times):
        z = z0
      elif times[i] < ascendStart:
        z = z0
      elif times[i] < sustainStart:
        z = z0 + (times[i] - ascendStart) * gain / ascend
      elif times[i] < descendStart:
        z = z0 + gain
      elif times[i] < restStart:
        z = z0 + gain - (times[i] - descendStart) * gain / descend

      out.append([x, y, z, h, a1, a2])
      #print(times[i] if i < len(times) else 'end', z)

    return out
