from sage.all import vector, var, point, list_plot
from random import randrange
import numpy as np

class Scenario:
    def __init__(self):
        # Array of R3 vectors
        self.users = []
    
    def addUser(self, user):
        self.users.append(user)
        
    # Add users randomly in a rectangle
    def addRandomGroundUsersUniform(self, n, xmin = -500, xmax = 500, ymin = -500, ymax = 500):
        for i in range(n):
            self.addUser(vector((
                randrange(xmin, xmax),
                randrange(ymin, ymax),
                0
            )))
        
    # Add users randomly in a circular region
    def addRandomGroundUsersUniformCircular(self, n, cx = 0, cy = 0, r = 500):
        r2 = r * r
        for i in range(n):
            dx = randrange(-r, r)
            dy = randrange(-r, r)
            while (dx * dx) + (dy * dy) > r2:
                dx = randrange(-r, r)
                dy = randrange(-r, r)
                
            self.addUser(vector((
                cx + dx,
                cy + dy,
                0
            )))
            
    def render(self, **kwargs):
        # Just render our ground users
        return sum([
            point(p, **kwargs) for p in self.users
        ])
            
    def evaluateCrafts(self, flights, nSamples = 1024, maxTime = None, loopFun = lambda x, **kwargs: x):
        # flight is an array of Flight
        # loopFun could be e.g. tqdm to track progress
        
        # XXX in LTE anything nearer than this is the same throughput
        if maxTime is None:
            # Assume that our total time is the length of the longest trajectories flight
            maxTime = max([flight.cycleTime for flight in flights])
            
        times = [i / nSamples * maxTime for i in range(nSamples)]
        
        result = np.zeros((len(flights), len(self.users), nSamples))
        
        px, py, pz = var('px, py, pz')
        
        
        for fidx in loopFun(range(len(flights))):
            flight = flights[fidx]
            minDist = 1500
            minDistSq = minDist ** 2
            dSq = var('dSq')
            maxThru = flight.R(dSq = minDistSq).n()
            
            # So these things are slow...
            dSquared, R, time_end = flight.getFasterEvaluatable(times[0])
            
            # Presume we're using tqdm and remove inner bar
            # Probably doesn't have any other impact...
            for tidx in loopFun(range(nSamples), leave = False):
                t = times[tidx]
                
                # Update our fast-callable fxn
                if t >= time_end:
                    dSquared, R, time_end = flight.getFasterEvaluatable(t)
                
                for uidx in range(len(self.users)):
                    ux, uy, uz = self.users[uidx]
                
                    d2 = dSquared(t, ux, uy, uz)
                    if d2 < minDistSq:
                        result[fidx, uidx, tidx] = maxThru
                    else:
                        result[fidx, uidx, tidx] = R(d2)
        
        return times, result

                  
    def plotResults(self, times, result):
        nSamples = len(times)
        
        means = np.mean(result[0,:,:], 0) / 1e6
        mins = np.min(result[0,:,:], 0) / 1e6
        maxs = np.max(result[0,:,:], 0) / 1e6
        
        plotMeans = [(times[i], means[i]) for i in range(nSamples)]
        plotMins = [(times[i], mins[i]) for i in range(nSamples)]
        plotMaxs = [(times[i], maxs[i]) for i in range(nSamples)]
        
        return sum([
            list_plot(plotMeans, plotjoined=True, color='red'),
            list_plot(plotMins, plotjoined=True, color='red'),
            list_plot(plotMaxs, plotjoined=True, color='red')
        ])