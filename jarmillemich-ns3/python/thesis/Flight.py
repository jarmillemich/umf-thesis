from sage.all import piecewise, parametric_plot, var, n, log, fast_callable, RDF

def dB(v):
    return n(10 * log(v, 10))

def dB2W(dB):
    return 10**(dB / 10)

# Represents a craft, trajectory, and alpha values for the trajectory
# Also, our transmitter specifications
class Flight:
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
            raise IndexError('Flight trajectory and alphas arguments must match length')
        
        self._craft = craft
        self._trajectory = trajectory
        self._alphas = alphas
        
        # Compute some things
        self._calcVelocityThrustPower()
        self._calcTotalTime()
        #self._createPositionFunctions()
        
        sigmaSquared = N0 + dB(B)
        self.B = B
        self.gamma = dB2W(B0 + xmitPower - sigmaSquared)
        
        #self._createBandwidthFunctions()
        
    def _calcVelocityThrustPower(self):
        self.vtp = []
        for i in range(len(self._trajectory.pieces)):
            piece = self._trajectory.pieces[i]
            alpha = self._alphas[i]
            
            self.vtp.append(piece.velocityThrustPower(self._craft, alpha))
        
    def _calcTotalTime(self):
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
        t, px, py, pz, dSq = var('t, px, py, pz, dSq')
        self.dSquared = (self.px - px)**2 + (self.py - py)**2 + (self.pz - pz)**2
        self.R = self.B * log(1 + self.gamma / dSq, 2)
        
    # Gets a faster version of our bandwidth function
    # In conjunction with other code, eliminates the overhead needed to
    # demux the piecewise function, except on inter-piece boundaries
    # Also wraps the inner function in fast_callable
    # def getFastEvaluatable(self, t0):
    #     # Find the right domain
    #     domain_idx = None
        
    #     for i in range(len(self.px.domains())):
    #         dom = self.px.domains()[i]
    #         if t0 >= dom.inf() and t0 < dom.sup():
    #             domain_idx = i
    #             break
                
    #     if domain_idx is None:
    #         raise RuntimeError('Not in domain')
            

    #     t, px, py, pz, dSq = var('t, px, py, pz, dSq')
        
    #     # Have to rebuild R, cannot get pieces out of a wrapped piecewise
    #     at_px = self.px.expressions()[domain_idx]
    #     at_py = self.py.expressions()[domain_idx]
    #     at_pz = self.pz.expressions()[domain_idx]
        
    #     dSquared = (at_px - px)**2 + (at_py - py)**2 + (at_pz - pz)**2
        
        
    #     R = self.B * log(1 + self.gamma / dSq, 2).function(t, dSq)
        
    #     dSquared = fast_callable(dSquared, vars=[t, px, py, pz], domain=float)
    #     func = fast_callable(R, vars=[dSq], domain=float)
        
    #     domain_end = self.px.domains()[domain_idx].sup()
    #     return dSquared, func, domain_end
    
    def getFasterEvaluatable(self, t0):
        t_at = 0
        
        for i in range(len(self._trajectory.pieces)):
            piece = self._trajectory.pieces[i]
            alpha = self._alphas[i]
            #v, t, p = piece.velocityThrustPower(self._craft, alpha)
            v, t, p = self.vtp[i]
            t_at += piece.length / v
            
            if t_at > t0:
                # This is the piece
                posFunc = piece.fastPiece(t0, v)
                
                def dSquared(t, px, py, pz):
                    ax, ay, az = posFunc(t)
                    return (px - ax)**2 + (py - ay)**2 + (pz - az)**2
                
                def R(dSq):
                    return self.B * log(1 + self.gamma / dSq, 2)
                
                return dSquared, R, t_at
            
            
            
        raise TypeError('Fell through')
        
    def render(self, **kwargs):
        t = var('t')
        return parametric_plot([self.px, self.py, self.pz], (t, 0, self.cycleTime - 0.00001), **kwargs)
    
    def toSim(self):
        from ns.mobility import PathMobilityModel
        
        model = PathMobilityModel()
        for i in range(len(self._trajectory.pieces)):
            piece = self._trajectory.pieces[i]
            alpha = self._alphas[i]
            model.AddSegment(piece.toSim(self._craft, alpha))

        return model
    
    # Generate the poses for the specified times
    def toPoses(self, times):
        import pandas as pd
        import numpy as np
        
        
        t0 = times[0]

        ret = []
        idx = 0
        endTime = times[-1]
        print(endTime)

        #for i in range(len(self._trajectory.pieces)):
        while t0 < endTime:
            piece = self._trajectory.pieces[idx]
            alpha = self._alphas[idx]
            v, t, p = self.vtp[idx]

            #print(t0, piece)
            t0, posePiece = piece.toPoses(times, t0, v, alpha)
            power = pd.Series([p for t in posePiece.index], dtype=float, index=posePiece.index)
            posePiece.insert(0, 'power', power, True)
            ret.append(posePiece)
            idx += 1
            idx %= len(self._trajectory.pieces)

        return pd.concat(ret)
