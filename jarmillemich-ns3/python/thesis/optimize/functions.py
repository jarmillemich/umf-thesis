# Just our optimization helpers
import numpy as np

class Defunc:
    """For constructing expressions"""
    def __init__(self, eval, debugString = None):
        self.eval = eval
        self.debugString = debugString

    def __call__(self, state):
        """Evaluate the expression we represent"""
        return self.eval(state)

    # Make this pretty looking
    def debug(self, state):
        """Build out a string to see what is being evaluated"""
        if self.debugString is None:
            return str(self(state))
        elif isinstance(self.debugString, list):
            out = []
            for item in self.debugString:
                if isinstance(item, Defunc):
                    out.append(str(item.debug(state)))
                elif item is None:
                    out.append(str(self(state)))
                else:
                    out.append(str(item))

            return ''.join(out)
        else:
            return self.debugString

    def _sanitize(self, other):
        """Try and handle constants"""
        ret = other

        if type(other) is float or type(other) is int:
            ret = Defunc(lambda state: other)
        

        if not isinstance(ret, Defunc):
            raise TypeError('Defunc builder second arg is not a Defunc: ' + str(ret))

        return ret

    def __add__(self, other):
        other = self._sanitize(other)
        return Defunc(lambda state: self.eval(state) + other.eval(state), ['(', self, '+', other, ')'])


    def __sub__(self, other):
        other = self._sanitize(other)
        return Defunc(lambda state: self.eval(state) - other.eval(state), ['(', self, '-', other, ')'])

    def __rsub__(self, other):
        other = self._sanitize(other)
        return other - self

    def __mul__(self, other):
        other = self._sanitize(other)
        return Defunc(lambda state: self.eval(state) * other.eval(state), ['(', self, '*', other, ')'])

    def __rmul__(self, other):
        other = self._sanitize(other)
        return other * self

    def __truediv__(self, other):
        other = self._sanitize(other)
        return Defunc(lambda state: self.eval(state) / other.eval(state), ['(', self, '/', other, ')'])

    # Sum-thing awfulness
    def __radd__(self, other):
        if other == 0:
            return self

        other = self._sanitize(other)

        return other + self


def radiusPenalty(radius, degree = 2):
    """Penalize encroachment on the surrounding regions"""
    import numpy as np
    def inner(stats):
        poses = stats['poses']
        dist = np.sqrt(poses.x**2 + poses.y**2)
        violation = dist - radius
        violation[violation < 0] = 0
        return -sum(violation**degree)

    return Defunc(inner, ['RP<', None, '>'])

def altitudePenalty(lo, hi, degree = 2):
    """Penalize encroachment on volumes above/below our volume"""
    def inner(stats):
        poses = stats['poses']
        above = poses.z - hi
        below = lo - poses.z
        above[above < 0] = 0
        below[below < 0] = 0
        violation = above + below
        return -sum(violation**degree)

    return Defunc(inner, ['AP<', None, '>'])

def calcBatteryChange(stats):
    battery = stats['battery']
    ts, tf = battery.index[0], battery.index[-1]
    return battery[tf] - battery[ts]

def calcGravityChange(stats):
    poses = stats['poses']

    ts, tf = poses.index[0], poses.index[-1]
    dz = poses.z[tf] - poses.z[ts]
    mass = stats['mass']
    g = 9.8 # Technically changes by a small amount, 9.776-9.805 in our range
    #poses.z.plot()
    #print(mass, g, dz)
    return mass * g * dz / 3600

def batteryReward():
    """Reward increasing energy"""
    return Defunc(calcBatteryChange, ['BR<', None, '>'])

def gravityReward():
    """Reward stored gravitational energy"""
    return Defunc(calcGravityChange, ['GR<', None, '>'])


def energyPenalty(budget, gravityCoeff = 0.5, degree=2):
    """
    Penalize exceeding our energy budget (Wh!), including battery and gravity

    budget > 0 means we can lose energy and meet budget
    budget < 0 means we must harvest energy to meet budget
    """
    def inner(stats):
        E_g = calcGravityChange(stats)
        E_b = calcBatteryChange(stats)

        E_delta = E_g * gravityCoeff + E_b

        deficit = -budget - E_delta
        if deficit < 0: deficit = 0

        return -(deficit**degree)

    return Defunc(inner, ['EP<', None, '>'])

def batteryPenalty(threshold, degree = 2):
    """Penalize going below a certain charge threshold"""
    def inner(stats):
        battery = stats['battery']
        defecit = threshold - battery
        defecit[defecit < 0] = 0
        return -(defecit.sum()**degree)

    return Defunc(inner, ['BP<', None, '>'])

def throughputReward(weights = None):
    """
    Reward higher levels of available throughput

    Optionally, weight some users above others
    """
    def inner(stats):
        thru = stats['throughput']

        rates = thru.mean(axis=0)
        if weights is not None:
            rates *= np.array(weights)

        # TODO Is this the best way?
        return rates.sum()

    return Defunc(inner, ['TR<', None, '>'])

def throughputPenalty(levels, degree = 2):
    """
    Penalize under-serving the given levels

    Specifically, penalize deficit in mean over window
    """
    # TODO Maybe we should do min instead?
    def inner(stats):
        thru = stats['throughput']
        rates = thru.mean(axis=0)
        violation = levels - rates
        violation[violation < 0] = 0

        return -sum(violation**degree)

    return Defunc(inner, ['TP<', None, '>'])


def thrustPenalty(hi, penalty = 999):
    """Discourage using higher than available thrust"""
    def inner(stats):
        thrust = stats['poses'].thrust
        delta = np.zeros(len(thrust))
        delta[thrust > hi] = 1
        delta *= penalty

        return -sum(delta)

    return Defunc(inner, ['TRP<', None, '>'])

def alphaPenalty(lo = -5, hi = 12, penalty = 999):
    """
    Penalize high/low angles of attack.
    
    If we go outside our Angle-of-Attack limits
    We start being in non-aerodynamic/rocket mode, which our
    equations do NOT work for (hovering reports 0 power use)
    """
    def inner(stats):
        alpha = stats['poses'].alpha
        delta = np.zeros(len(alpha))
        delta[alpha < lo] = 1
        delta[alpha > hi] = 1
        delta *= penalty

        return -sum(delta)

    return Defunc(inner, ['AP<', None, '>'])

def speedPenalty(lo, hi, degree = 2):
    """Penalize going to fast/slow"""
    def inner(stats):
        poses = stats['poses']
        above = poses.v - hi
        below = lo - poses.v
        above[above < 0] = 0
        below[below < 0] = 0
        violation = above + below
        return -sum(violation**degree)

    return Defunc(inner, ['VP<', None, '>'])



def makeZSchedule(gain, schedule, time):
    """
    Make a Z schedule over the specified times (seconds from start!)
    
    Gain is distance from rest altitude to peak altitude.
    Schedule is (rest, ascend, sustain, descend) times, in seconds.
    Rest is actually the rest before ascent, and does not include the rest after descent.
    Will not work for more than one cycle
    """
    time = np.array(time)
    
    rest, ascend, sustain, descend = schedule
    ascendStart = rest
    sustainStart = ascendStart + ascend
    descendStart = sustainStart + sustain
    restStart = descendStart + descend
    
    out = np.zeros(len(time))
    
    # Descend
    out[time < restStart] = gain - (time[time < restStart] - descendStart) * gain / descend
    # Sustain
    out[time < descendStart] = gain
    # Ascend
    out[time < sustainStart] = (time[time < sustainStart] - ascendStart) * gain / ascend
    # rest
    out[time < rest] = 0
    
    return out

# Builds a fitness function: vec |=> R
# expr: fitness expression. If a list, will be summed
# times: pd.date_range of times to use
# dt: Time granularity. (Currently automatically determined by # of segments provided and total duration)
from thesis.Trajectory import ImperativeTrajectory
from thesis.trajectory.SplineyTrajectory import SplineyTrajectory
from thesis.Flight import Flight

class FitnessHelper:
    def __init__(self,
                 judge,
                 craft,
                 times,
                 expr = None,
                 maxBank = 10,
                 startPos = (0, 0, 1000),
                 startHeading = 0,
                 useZSchedule = False
    ):
        if isinstance(expr, list):
            expr = sum(expr)

        if not isinstance(expr, Defunc):
            print(expr)
            raise TypeError('Please use a Defunc fitness, got ' + str(expr))

        self.judge = judge
        self.craft = craft
        self.expr = expr
        self.times = times
        self.totalTime = (self.times[-1] - self.times[0]).total_seconds()
        self.useZSchedule = useZSchedule
        self.maxBank = maxBank

        self.startPos = startPos
        self.startHeading = startHeading

    def getTrajBuilder(self):
        import numpy as np

        def mapAltitude(times, zSchedule = None):
            # Nope, you get FULL control
            if not self.useZSchedule: return times * 0

            return makeZSchedule(zSchedule[0], zSchedule[1:], times)

        def clamp(v, l = 0, h = 1):
            return max(min(v, h), l)

        def vectorToTrajectory(vec):
            if self.useZSchedule:
                zSchedule = vec[:5]
                vec = vec[5:]
            else:
                zSchedule = None
            
            codonsPerSegment = 3
            alphas = vec[0::codonsPerSegment] 
            banks = vec[1::codonsPerSegment]
            zOffsets = vec[2::codonsPerSegment]

            # Enforce starting/ending altitude
            # zOffsets[0] = 0
            # zOffsets[-1] = 0
            
            nSegments = len(vec) // codonsPerSegment
            
            # + 1 for luck (AKA fix numerical error)
            dt = (self.totalTime + 1) / nSegments

            seconds = np.arange(0, self.totalTime, dt)
            altitudes = mapAltitude(seconds, zSchedule) + self.startPos[2]


            #print(altitudes)
            
            coords = []
            #alphas = []
            lastZ = self.startPos[2]
            for i in range(nSegments):
                zOffset = zOffsets[i]
                if self.useZSchedule:
                    # Z is relative to our schedule
                    nextZ = altitudes[i] + zOffset
                else:
                    # Z is relative to wherever we are
                    nextZ = lastZ + zOffset
                dz = nextZ - lastZ
                lastZ = nextZ

                bank = banks[i]
                
                coords.append([
                    alphas[i],
                    clamp(bank, -self.maxBank, self.maxBank),
                    dz,
                ])
                
                #alphas.append(5)
                
            traj = ImperativeTrajectory(
                self.craft,
                self.startPos,
                self.startHeading,
                coords,
                commandDuration=dt
            )
            return traj, alphas

        return vectorToTrajectory

    def getFitness(self, trajBuilder = None, debug = False, initial_charge = 0.1):
        if trajBuilder is None:
            trajBuilder = self.getTrajBuilder()

        # TODO make configurable
        
        # From NS3 defaults
        radioParams = {
            'xmitPower': 30, # dBm
            'B': 180e3 * 25, # 25 180kHz RBs = 4.5 MHz
            'N0': -174       # See lte-spectrum-value-helper.cc kT_dBm_Hz
        }

        # Keep track of how many fitness evaluations he have done
        # (Possibly not thread safe, but an appx value is fine)
        import multiprocessing
        cntr = multiprocessing.Value('i', 0)

        def inner(vec):
            cntr.value += 1
            traj, alphas = trajBuilder(vec)
            flight = Flight(self.craft, traj, alphas, **radioParams)
            
            # XXX This initial charge assumes we start in high-solar times
            stats = self.judge.flightStats(flight, times=self.times, initial_charge=initial_charge)

            if debug:
                return self.expr.debug(stats)
            else:
                return self.expr(stats)
            
        inner.evaluations = cntr

        return inner

class SplineyFitnessHelper(FitnessHelper):
    """
    FitnessHelper but with our SplineyTrajectory instead of ImperativeTrajectory
    """
    def __init__(self, *args, desiredDuration = None, initialPosition = None, zMode = 'absolute', **kwargs):
        self.desiredDuration = desiredDuration
        self.initialPosition = initialPosition
        self.zMode = zMode
        super().__init__(*args, **kwargs)

    def getTrajBuilder(self):
        import numpy as np

        def mapAltitude(times):
            # times should be an np.array of seconds
            # Lerp the altitude between zMin and zMax over our time range
            dt = times[-1] - times[0]
            at = times / dt
            alt = at * (zMax - zMin) + zMin
            return alt

        def clamp(v, l = 0, h = 1):
            return max(min(v, h), l)

        def vectorToTrajectory(vec):
            # X Y Z Heading alpha1 alpha2
            codonsPerSegment = 6
            def chunks(lst, n):
                """Yield successive n-sized chunks from lst."""
                for i in range(0, len(lst), n):
                    yield lst[i:i + n]
            
            zSchedule = None

            if self.zMode == 'delta':
                zOffsets = np.array(vec[2::codonsPerSegment])
                zOffsets = zOffsets.cumsum() + 1000 # TODO not hard code
                # Loop back around
                zOffsets[-1] = zOffsets[0]
                print(zOffsets)
                vec[2::codonsPerSegment]
            if self.zMode == 'schedule':
                codonsPerSegment = 5
                zSchedule, vec = vec[:6], vec[6:]

            waypoints = list(chunks(vec, codonsPerSegment))

            if self.initialPosition is not None:
                waypoints[0][:len(self.initialPosition)] = self.initialPosition

            

            fixedFirst = self.initialPosition is not None
                
            traj = SplineyTrajectory(
                waypoints,
                desiredDuration=self.desiredDuration,
                craft=self.craft,
                minimumRadius = 20,
                fixedFirst = fixedFirst,
                zSchedule = zSchedule
            )
            alphas = traj.alphas
            return traj, alphas

        return vectorToTrajectory