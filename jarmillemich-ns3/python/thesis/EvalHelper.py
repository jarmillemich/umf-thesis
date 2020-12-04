from thesis.Trajectory import Trajectory
from thesis.Flight import Flight
from thesis.Genetics import Chromosome
import numpy as np


# Various things to help generate and evaluate candidate trajectories
class Judge:
    def __init__(self,
                 scene,
                 craft,
                 xRange = (-5000, 5000),
                 yRange = (-5000, 5000),
                 zRange = (1000, 2000),
                 radiusRange = (0, 1000),
                 alphaRange = (1, 10),
                 numberOfSegments = 8,
                 # False is 8 bits, True is 16
                 largerParameters = False,
                 # If we generate a radius less than this, don't emit the circle
                 circleSuppressionLimit = 100
                ):
        self._scene = scene
        self._craft = craft
        
        self._xRange = xRange
        self._yRange = yRange
        self._zRange = zRange
        self._radiusRange = radiusRange
        self._alphaRange = alphaRange
        self._numberOfSegments = numberOfSegments
        self._largerParameters = largerParameters
        self._circleSuppressionLimit = circleSuppressionLimit
        
    def newChromosome(self):
        bits = 16 if self._largerParameters else 8
        parmsPerSegment = 4
        nAlphas = self._numberOfSegments
        nParams = self._numberOfSegments * parmsPerSegment + nAlphas
        
        return Chromosome(nParams * bits)
        
    def generateChromosome(self, chromo):
        bits = 16 if self._largerParameters else 8
        parmsPerSegment = 4
        nAlphas = self._numberOfSegments
        nParams = self._numberOfSegments * parmsPerSegment + nAlphas
        
        xMin, xMax = self._xRange
        yMin, yMax = self._yRange
        zMin, zMax = self._zRange
        rMin, rMax = self._radiusRange
        aMin, aMax = self._alphaRange
        
        def gp(n, l, h):
            if bits == 8:
                return chromo.getReal8(n * bits, lower=l, upper=h)
            elif bits == 16:
                return chromo.getReal8(n * bits, lower=l, upper=h)
            else:
                raise IndexError('not the right number of bits')

        def gx(n):
            return gp(6 * n + 0, xMin, xMax)

        def gy(n):
            return gp(6 * n + 1, yMin, yMax)

        def gz(n):
            return gp(6 * n + 2, zMin, zMax)

        def gr(n):
            return gp(6 * n + 3, rMin, rMax)

        def ga(n, i):
            return gp(6 * n + 4 + i, aMin, aMax)

        circles = [
            [gx(i), gy(i), gr(i), gz(i)]
            for i in range(self._numberOfSegments)
            # Have the ability to suppress extra circles??
            if gr(i) > self._circleSuppressionLimit
        ]    

        alphas = [
            ga(n, i)
            for n in range(self._numberOfSegments)
            for i in [0, 1]
            if gr(n) > self._circleSuppressionLimit
        ]

        return circles, alphas
        
    def judgeChromosome(self, chromo, dbg = False):
        flight = self.chromosomeToFlight(chromo)
        return self.judgeFlight(flight, dbg = dbg)
    
    def chromosomeToFlight(self, chromo):
        wayCircles, alphas = self.generateChromosome(chromo)
        
        trajectory = Trajectory(wayCircles)

        flight = Flight(
            self._craft,
            trajectory,
            alphas,
            # TODO bring these in from outside...
            # From https://www.doubleradius.com/site/stores/baicells/baicells-nova-233-gen-2-enodeb-outdoor-base-station-datasheet.pdf
            #xmitPower = 30,
            #B = 5e6
            # From https://yatebts.com/products/satsite/
    #         xmitPower = 43,
    #         B = 5e6,
            # From the Zeng paper
            #xmitPower = 10,
            #B = 1e6
            # To match NS3 defaults
             xmitPower = 30,
             B = 180e3 * 25, # 25 180kHz Resource Blocks, = 4.5 MHz
            # See lte-spectrum-value-helper.cc kT_dBm_Hz
            N0 = -174,
            
        )
        return flight
        
    def judgeFlight(self, flight, dbg = False):
        times, result = self._scene.evaluateCrafts([flight])

        if dbg:
            # The ridiculous size is needed for Sage 9.0
            positions = (flight._trajectory.render() + self._scene.render(size=20000))#.scale(0.001)
        else:
            positions = None

        # Mean flight power in watts
        meanFlightPower = flight.cycleEnergy / flight.cycleTime
        # Minimum throughput of any user in Mbps
        thru = np.min(result) / 1e6
        # Roughly Mb/J
        score = (thru / meanFlightPower).n()

        if dbg:
            from sage.all import plot, var
            thruPlot = self._scene.plotResults(times, result) + plot([thru], (var('x'), min(times), max(times)), ymin = 0)
        else:
            thruPlot = None

        return score, thru, meanFlightPower, flight.cycleTime, positions, thruPlot