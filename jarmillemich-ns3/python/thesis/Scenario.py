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

    def posesToThroughput(self, flight, poses):
        
        # Convert some poses to a dataframe of throughputs in bps
        #idx = pd.MultiIndex.from_product([poses.index, [i for i in range(len(self.users))]], names=['time', 'user'])
        #frame = pd.DataFrame(index=idx, columns=['throughput'], dtype='float64')

        B = flight.B
        gamma = flight.gamma

        thru = []

        x = np.array(poses.x)
        y = np.array(poses.y)
        z = np.array(poses.z)

        for i in range(len(self.users)):
            ux, uy, uz = self.users[i]
            
            dx = x - ux
            dy = y - uy
            dz = z - uz
            
            distSq = dx ** 2 + dy ** 2 + dz ** 2
            R = B * np.log2(1 + gamma / distSq)
            
            #print(i, len(frame.throughput[:,i]), len(R))
            #frame.throughput.loc[:,i] = R
            thru.append(R)

        frame = np.transpose(thru)
        #datas = pd.DataFrame(np.transpose(thru)).stack()
        #frame = pd.DataFrame(np.transpose(thru), dtype='float64')

        # LTE does not improve if closer than this distance
        minDist = 1500
        minDistSq = minDist ** 2
        minDistR = B * np.log2(1 + gamma / minDistSq)
        frame[frame > minDistR] = minDistR
            
        return frame

    def posesToThroughputFrame(self, flight, poses):
        # The same as the non-frame version, but returns a indexed DataFrame (SLOWER)
        import pandas as pd
        data = self.posesToThroughput(flight, poses)

        idx = pd.MultiIndex.from_product([poses.index, [i for i in range(len(self.users))]], names=['time', 'user'])
        return pd.DataFrame(pd.DataFrame(data).stack(), columns=['throughput'], index=idx, dtype='float64')
                  
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