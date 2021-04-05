from sage.all import piecewise, parametric_plot, var, n, log, fast_callable, RDF

def dB(v):
    """Convert linear value to dB scale."""
    return n(10 * log(v, 10))

def dB2W(dB):
    """Convert dB value to linear scale."""
    return 10**(dB / 10)

class Flight:
    """Represents a craft, trajectory, and alpha values for the trajectory, as well as the transmit parameters."""
    def __init__(self,
                 craft,
                 trajectory,
                 alphas,
                 # Hertz
                 B = 1e6,
                 # dBm - transmission power
                 xmitPower = 10,
                 # dBm/Hz - noise power spectrum density
                 N0 = -170,
                 # dB - reference channel power
                 B0 = -50
                ):
        if len(trajectory.pieces) != len(alphas):
            raise IndexError('Flight trajectory and alphas arguments must match length %s != %s' % (len(trajectory.pieces), len(alphas)))
        
        self._craft = craft
        self._trajectory = trajectory
        self._alphas = alphas
        
        # Pre-compute some items
        self._calcVelocityThrustPower()
        self._calcTotalTime()
        # Extremely slow, only use when needed
        #self._createPositionFunctions()
        
        sigmaSquared = N0 + dB(B)
        self.B = B
        self.gamma = dB2W(B0 + xmitPower - sigmaSquared)
        
        # Also extremely slow
        #self._createBandwidthFunctions()
        
    def _calcVelocityThrustPower(self):
        """Pre-compute the velocity, thrust, and power for each of our segments."""
        self.vtp = []
        for i in range(len(self._trajectory.pieces)):
            piece = self._trajectory.pieces[i]
            alpha = self._alphas[i]
            
            self.vtp.append(piece.velocityThrustPower(self._craft, alpha))
        
    def _calcTotalTime(self):
        """Pre-compute the total cycle time and energy used."""
        # Compute the cycle time of this flight
        totalTime = 0
        
        # Compute the total cycle energy as well
        totalEnergy = 0
        
        for i in range(len(self._trajectory.pieces)):
            segment = self._trajectory.pieces[i]
            alpha = self._alphas[i]
            
            length = segment.length
            vel, thr, pwr = self.vtp[i]
            
            dt = length / vel
            totalTime += dt
            
            totalEnergy += dt * abs(pwr)
            
        self.cycleTime = totalTime
        self.cycleEnergy = totalEnergy
        
    def _createPositionFunctions(self):
        """Create piecewise functions for our position for each segment (SLOW!)."""
        t_at = 0
        pieces = []
        
        for i in range(len(self._trajectory.pieces)):
            segment = self._trajectory.pieces[i]
            alpha = self._alphas[i]
            
            #vel, thr, pwr = segment.velocityThrustPower(self._craft, alpha)
            vel, thr, pwr = self.vtp[i]
            #print('pre t_at=%s, vel=%s, thr=%s, pwr=%s, alpha=%s' % (t_at, vel, thr, pwr, alpha))
            time, func = segment.piece(t_at, vel)
            #print('time after', time)
            t_at = time.sup()
            pieces.append((time, func))
            
        self.px = piecewise([(time, func[0]) for time, func in pieces])
        self.py = piecewise([(time, func[1]) for time, func in pieces])
        self.pz = piecewise([(time, func[2]) for time, func in pieces])
        
    def _createBandwidthFunctions(self):
        """Create piecewise bandwidth functions after computing position functions (SLOW!)."""
        t, px, py, pz, dSq = var('t, px, py, pz, dSq')
        self.dSquared = (self.px - px)**2 + (self.py - py)**2 + (self.pz - pz)**2
        self.R = self.B * log(1 + self.gamma / dSq, 2)
                
    def render(self, **kwargs):
        """Render based on position functions."""
        t = var('t')
        return parametric_plot([self.px, self.py, self.pz], (t, 0, self.cycleTime - 0.00001), **kwargs)
    
    def toSim(self):
        """Convert flight into the NS3 mobility model for simulation."""
        from ns.mobility import PathMobilityModel
        
        model = PathMobilityModel()
        for i in range(len(self._trajectory.pieces)):
            piece = self._trajectory.pieces[i]
            alpha = self._alphas[i]
            model.AddSegment(piece.toSim(self._craft, alpha))

        return model
    
    def toPoses(self, times):
        """Generate the poses for the specified times."""
        import pandas as pd
        import numpy as np
        
        
        t0 = times[0]
        dimes = np.array([t.total_seconds() for t in times - t0])

        ret = []
        idx = 0
        endTime = (times[-1] - t0).total_seconds()

        t_at = 0

        while t_at < endTime:
            piece = self._trajectory.pieces[idx]
            alpha = self._alphas[idx]
            v, thr, p = self.vtp[idx]

            #print(t0, piece)
            dt, posePiece = piece.toPosesTest(dimes, t_at, v, alpha, thrust=thr, power=p, craft=self._craft)
            t_at += dt
            ret.append(posePiece)
            idx += 1
            idx %= len(self._trajectory.pieces)

        data = np.concatenate(ret, axis=1).transpose()
        return pd.DataFrame(data, columns=['x','y','z','v','tilt','azimuth','thrust','power','alpha'], index=times)

