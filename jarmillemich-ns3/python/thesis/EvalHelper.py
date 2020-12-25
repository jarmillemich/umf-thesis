from thesis.Trajectory import WaycircleTrajectory
from thesis.Flight import Flight
from thesis.Genetics import Chromosome
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sage.all import point, hue, line, show


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
        
        trajectory = WaycircleTrajectory(wayCircles)

        flight = Flight(
            self._craft,
            trajectory,
            alphas,
            # TODO bring these in from outside...
            # From https://www.doubleradius.com/site/stores/baicells/baicells-nova-233-gen-2-enodeb-outdoor-base-station-datasheet.pdf
            #xmitPower = 30,
            #B = 5e6
            # From https://yatebts.com/products/satsite/
            # xmitPower = 43,
            # B = 5e6,
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

    # Some info on trajectory
    def displayFlightTrajectoryInfo(self, flight, render = True, threed = False):
        T0 = pd.to_datetime('2020-07-01T00')
        dt = pd.to_timedelta(int(flight.cycleTime), 'S')
        times = pd.date_range(start=T0, end=T0 + dt, freq='30S', tz='America/Detroit')

        print(' Flight Info')
        print('=' * 80)

        print('%d Second cycles, for %d cycles with %d left over' % (
            flight.cycleTime,
            (24 * 3600) / flight.cycleTime,
            (24 * 3600) % flight.cycleTime
        ))
        poses = flight.toPoses(times.to_series())

        print('altitude min=%d mean=%d max=%d' % (poses.z.min(), poses.z.mean(), poses.z.max()))


        if render:
            if threed:
        #         show(sum([
        #             line(((
        #                 poses['x'][poses.index[i-1]],
        #                 poses['y'][poses.index[i-1]],
        #                 poses['z'][poses.index[i-1]]
        #             ), (
        #                 poses['x'][poses.index[i]],
        #                 poses['y'][poses.index[i]],
        #                 poses['z'][poses.index[i]]
        #             )), color=hue(i/len(poses.index)), linewidth=400)
        #             for i in range(len(poses.index))
        #         ]))
                show(flight._trajectory.render())
            else:
                points = [
                    point((
                        poses['x'][poses.index[i]],
                        poses['y'][poses.index[i]],
                        #poses['z'][poses.index[i]]
                    ), color=hue(i / len(poses.index)), size=100)
                    for i in range(len(poses.index))
                ]
                points.extend([
                    point((x, y))
                    for x, y, z
                    in self._scene.users
                ])
                show(sum(points), aspect_ratio=1)

        print()
        print()
        
    def displayFlightPower(self, flight, bat_Wh_cap, P_payload, start = '2020-07-01T08', end = '2020-07-03T08'):
        print(' Power info')
        print('=' * 80)
        
        times = pd.date_range(start=start, end=end, freq='30S', tz='America/Detroit')
        poses = flight.toPoses(times.to_series())
        
        # TODO this is an arbitrary location near UMF
        solar = flight._craft.calcSolarPower(poses, 43, -84)
        battery = flight._craft.calcBatteryCharge(poses, solar, bat_Wh_cap, constant_draw = P_payload)

        fix, ax1 = plt.subplots(figsize=(18,10))
        ax1.set_xlabel('local time')
        ax1.set_ylabel('Power [W]')
        ax1.set_ylim(bottom = 0, top = solar.max() * 1.1)

        solar.plot(color='tab:orange')
        poses.power.plot(color='tab:blue')
        (poses.power + P_payload).plot(color='tab:cyan')

        # Gravitational potential energy in Wh
        base_height = poses['z'].min()
        E_g = (poses.z - base_height) * 9.8 * flight._craft._mass / 3600

        ax2 = ax1.twinx()
        ax2.set_ylabel('Energy [Wh]', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        ax2.set_ylim(bottom = 0, top = (battery + E_g).max() * 1.1)
        battery.plot(color='tab:red')
        if poses['z'].max() - base_height > 10:
            # Show potential energy as well, if we have a reasonable amount
            (battery + E_g).plot(color='tab:cyan')

        # Note, we take after 6 hours to get past the initial charge
        mSocStart = pd.to_datetime(start) + pd.offsets.Hour(6)
        print('mSoc = %.2f%%' % (battery[mSocStart:].min() / bat_Wh_cap * 100))

        print()
        print()

    def displayFlightAltitudeThroughputInfo(self, flight, start = '2020-07-01T08', end = '2020-07-03T08'):
        print(' Throughput (estimated) and altitude')
        print('=' * 80)
        
        times = pd.date_range(start=start, end=end, freq='30S', tz='America/Detroit')
        poses = flight.toPoses(times.to_series())
        
        def calculateRates(user):
            ux, uy, uz = user
            dSq = (poses.x - ux)**2 + (poses.y - uy)**2 + (poses.z - uz)**2
            B, gamma = flight.B, flight.gamma

            R = B * np.log2(1 + gamma / dSq) / 1e6

            return R

        rates = {}

        for idx in range(len(self._scene.users)):
            user = self._scene.users[idx]
            rates['user-' + str(idx)] = calculateRates(user)

        agg = pd.DataFrame(rates)

        fix, ax1 = plt.subplots(figsize=(18,10))
        ax1.set_xlabel('local time')
        ax1.set_ylabel('Altitude')
        ax1.set_ylim(bottom = 0, top = poses.z.max() * 1.1)

        poses.z.plot(color='black')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Throughput [Mbps]', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        ax2.set_ylim(bottom = 0, top = agg.max().max() * 1.1)
        agg.min(1).rolling(window=int(10), center=True).mean().plot()
        agg.mean(1).rolling(window=int(10), center=True).mean().plot()
        agg.max(1).rolling(window=int(10), center=True).mean().plot()

        # Eh
        plt.figure(figsize=(18,10))
        poses.v.round(1).plot()

        # Gimpy 4 hour rolling mean?
        #agg.mean(1).rolling(window=int(4*2*60), center=True).mean().plot(figsize=(18,10))

        #return agg

    def runNS3Simulation(self, flight, start = '2020-07-01T08', end = None, burstArrivals = 1, burstLength = 1):
        # By default, run a single cycle
        if end is None:
            end = pd.to_datetime(start) + pd.offsets.Second(int(flight.cycleTime))

        dt = int((end - pd.to_datetime(start)).total_seconds())

        from ns.core import Vector, MilliSeconds, Seconds, StringValue
        from thesis.simulation import SimulationContext
        import ns.mobility
        from ns.mobility import PathMobilityModelSegments as Seg
        import ns.flow_monitor

        flh = ns.flow_monitor.FlowMonitorHelper()

        with SimulationContext() as sim:
            # Add our entities
            sim.addEnbFlight(flight)
            for user in self._scene.users:
                sim.addUser(*user)
            sim._finalizeNodes()
            
            for idx in range(len(self._scene.users)):
                # Create traffic generators
                ueNode = sim.ueNodes.Get(idx)
                ueNetDev = sim.ueLteDevs.Get(idx)
                ueIp = sim.ueIpIface.GetAddress(idx)
                
                sinkAddr = ns.network.InetSocketAddress(sim.remoteHostAddr, 9000 + idx)
                packetSinkHelper = ns.applications.PacketSinkHelper("ns3::UdpSocketFactory", sinkAddr)
                sim.serverApps.Add(packetSinkHelper.Install(sim.remoteHost))
                
                MbpsTarget = 1

                h = ns.applications.PPBPHelper("ns3::UdpSocketFactory", sinkAddr)
                # Supposed to be how many active "bursts" we have, but appears to be how many started per second
                h.SetAttribute("MeanBurstArrivals", StringValue("ns3::ConstantRandomVariable[Constant=%f]" % burstArrivals))
                # Duration in seconds of each burst
                h.SetAttribute("MeanBurstTimeLength", StringValue("ns3::ConstantRandomVariable[Constant=%f]" % burstLength))
                h.SetAttribute("BurstIntensity", StringValue("1Mb/s"))
                app = h.Install(ueNode)
                sim.clientApps.Add(app)
                
            # Set up traces
            
            
            sim.lteHelper.EnableRlcTraces()
            sim.lteHelper.EnablePhyTraces()
            
            
            flh.InstallAll()
            
            sim.startAndMonitorApps(resolution = 10)
            sim.stopAppsAt(dt)
            
            sim.run(dt + 10)
            
            return sim.serverAppStats, sim.clientAppStats