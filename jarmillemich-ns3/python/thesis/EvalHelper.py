from thesis.Trajectory import WaycircleTrajectory
from thesis.Flight import Flight
from thesis.Genetics import Chromosome
import numpy as np
import pandas as pd
from sage.all import point, hue, line, show


class Judge:
    """Various methods to help generate and evaluate candidate trajectories."""
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
        
    # def newChromosome(self):
    #     bits = 16 if self._largerParameters else 8
    #     parmsPerSegment = 4
    #     nAlphas = self._numberOfSegments
    #     nParams = self._numberOfSegments * parmsPerSegment + nAlphas
        
    #     return Chromosome(nParams * bits)
        
    # def generateChromosome(self, chromo):
    #     bits = 16 if self._largerParameters else 8
    #     parmsPerSegment = 4
    #     nAlphas = self._numberOfSegments
    #     nParams = self._numberOfSegments * parmsPerSegment + nAlphas
        
    #     xMin, xMax = self._xRange
    #     yMin, yMax = self._yRange
    #     zMin, zMax = self._zRange
    #     rMin, rMax = self._radiusRange
    #     aMin, aMax = self._alphaRange
        
    #     def gp(n, l, h):
    #         if bits == 8:
    #             return chromo.getReal8(n * bits, lower=l, upper=h)
    #         elif bits == 16:
    #             return chromo.getReal8(n * bits, lower=l, upper=h)
    #         else:
    #             raise IndexError('not the right number of bits')

    #     def gx(n):
    #         return gp(6 * n + 0, xMin, xMax)

    #     def gy(n):
    #         return gp(6 * n + 1, yMin, yMax)

    #     def gz(n):
    #         return gp(6 * n + 2, zMin, zMax)

    #     def gr(n):
    #         return gp(6 * n + 3, rMin, rMax)

    #     def ga(n, i):
    #         return gp(6 * n + 4 + i, aMin, aMax)

    #     circles = [
    #         [gx(i), gy(i), gr(i), gz(i)]
    #         for i in range(self._numberOfSegments)
    #         # Have the ability to suppress extra circles??
    #         if gr(i) > self._circleSuppressionLimit
    #     ]    

    #     alphas = [
    #         ga(n, i)
    #         for n in range(self._numberOfSegments)
    #         for i in [0, 1]
    #         if gr(n) > self._circleSuppressionLimit
    #     ]

    #     return circles, alphas
        
    # def judgeChromosome(self, chromo, dbg = False):
    #     flight = self.chromosomeToFlight(chromo)
    #     return self.judgeFlight(flight, dbg = dbg)
    
    # def chromosomeToFlight(self, chromo):
    #     wayCircles, alphas = self.generateChromosome(chromo)
        
    #     trajectory = WaycircleTrajectory(wayCircles)

    #     flight = Flight(
    #         self._craft,
    #         trajectory,
    #         alphas,
    #         # TODO bring these in from outside...
    #         # From https://www.doubleradius.com/site/stores/baicells/baicells-nova-233-gen-2-enodeb-outdoor-base-station-datasheet.pdf
    #         #xmitPower = 30,
    #         #B = 5e6
    #         # From https://yatebts.com/products/satsite/
    #         # xmitPower = 43,
    #         # B = 5e6,
    #         # From the Zeng paper
    #         #xmitPower = 10,
    #         #B = 1e6
    #         # To match NS3 defaults
    #          xmitPower = 30,
    #          B = 180e3 * 25, # 25 180kHz Resource Blocks, = 4.5 MHz
    #         # See lte-spectrum-value-helper.cc kT_dBm_Hz
    #         N0 = -174,
            
    #     )
    #     return flight
        
    def judgeFlight(self, flight, dbg = False):
        """Old version of flight fitness function."""
        #times, result = self._scene.evaluateCrafts([flight])

        # TODO define elsewhere
        bat_Wh_cap = 15.5 * 650

        times = pd.date_range(start = '2020-11-28T08', end = '2020-11-30T08', freq='120S', tz='America/Detroit').to_series()
        poses = flight.toPoses(times)
        throughput = self._scene.posesToThroughput(flight, poses)
        solar = flight._craft.calcSolarPower(poses, 36, -84)
        battery = flight._craft.calcBatteryCharge(poses, solar, 15.5*650, constant_draw = 109)

        #print('up to', poses.z.max())

        mSocStart = pd.to_datetime(times[0]) + pd.offsets.Hour(6)
        mSoc = battery[mSocStart:].min() / bat_Wh_cap * 100

        if dbg:
            # The ridiculous size is needed for Sage 9.0
            positions = (flight._trajectory.render() + self._scene.render(size=20000))#.scale(0.001)
        else:
            positions = None

        # # Mean flight power in watts
        meanFlightPower = flight.cycleEnergy / flight.cycleTime
        # # Minimum throughput of any user in kbps
        # thru = np.mean(result) / 1e3
        # # Roughly Mb/J
        # score = (thru / meanFlightPower).n()

        #meanFlightPower = poses.power.mean()
        thru = throughput.mean()[0]
        #score = thru / meanFlightPower / 1e3
        #score = thru * mSoc / 1e6
        score = mSoc

        if dbg:
            from sage.all import plot, var
            thruPlot = self._scene.plotResults(times, result) + plot([thru], (var('x'), min(times), max(times)), ymin = 0)
        else:
            thruPlot = None

        return score, thru, mSoc, flight.cycleTime, positions, thruPlot

    def flightStats(self, flight, times = None, bat_Wh_cap = 15.5 * 650, constant_draw = 109, extended = False, initial_charge = 0.5):
        """Compute flight statistics (poses, energy/power figures) for the given times and conditions."""
        import pandas as pd

        if times is None:
            times = pd.date_range(start = '2020-11-28T08', end = '2020-11-30T08', freq='120S', tz='America/Detroit').to_series()

        poses = flight.toPoses(times)
        throughput = self._scene.posesToThroughput(flight, poses)
        solar = flight._craft.calcSolarPower(poses, 36, -84)
        battery = flight._craft.calcBatteryCharge(poses, solar, bat_Wh_cap, constant_draw = constant_draw, initial_charge=initial_charge)

        

        ret = {
            # Some constants
            'mass': flight._craft._mass,
            
            'poses': poses, 
            'throughput': throughput,
            'solar': solar,
            'battery': battery,
            
        }

        if extended:
            # These are MEANINGLESS for a less-than-48-hour flight
            mSocStart = pd.to_datetime(times[0]) + pd.offsets.Hour(6)
            mSoc = battery[mSocStart:].min() / bat_Wh_cap * 100

            excessTime = flight._craft.calcExcessTime(battery)
            chargeMargin = flight._craft.calcChargeMargin(battery)

            ret['mSoc']= mSoc
            ret['excessTime'] = excessTime
            ret['chargeMargin'] = chargeMargin

        return ret
        

    def displayFlightTrajectoryInfo(self, flight, render = True, threed = False):
        """Display info and render trajectory."""
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
        
    def displayFlightPower(self, flight, bat_Wh_cap, P_payload, start = '2020-07-01T08', end = '2020-07-03T08', legend = False, savename=None):
        """Calculate and display/plot flight power statistics"""
        from matplotlib.dates import DateFormatter
        import matplotlib.pyplot as plt
        print(' Power info')
        print('=' * 80)
        
        times = pd.date_range(start=start, end=end, freq='30S', tz='America/Detroit')
        poses = flight.toPoses(times.to_series())
        
        # An arbitrary location near the top of NC/TN, USA
        solar = flight._craft.calcSolarPower(poses, 36, -84)
        print('sum solar influx:', solar.sum())
        battery = flight._craft.calcBatteryCharge(poses, solar, bat_Wh_cap, constant_draw = P_payload)

        fig, ax1 = plt.subplots()
        
        # Apparently this gets overwritten by the other, and is pointless
        #ax1.xaxis.set_major_formatter(DateFormatter('%H%M', tz='America/Detroit'))
        ax1.set_xlabel('Local Time [h]')
        ax1.set_ylabel('Power [W]')
        #ax1.set_ylim(bottom = 0, top = solar.max() * 1.1)
        ax1.set_ylim(bottom = 0, top = 5000)

        #solar.plot(color='tab:orange')
        ax1.plot(solar, color='orange')
        # NB Smoothing is for ladder trajectory, as the relatively rapid switches make it very hard to read
        poses.power.rolling(100, center=True).mean().plot(color='red', linestyle=(0,(2,4)))
        (poses.power + P_payload).rolling(64, center=True).mean().plot(color='darkgreen', linestyle=(0, (8,2)))

        
        # A quick look at the min/max power use
        totPower = (poses.power + P_payload).rolling(64, center=True).mean()
        print('Power range is %.2f to %.2f W' % (totPower.min(), totPower.max()))

        # Gravitational potential energy in Wh
        # TODO others apply a factor to this, as it is not a 100% equivalence to battery storage
        base_height = poses['z'].min()
        E_g = (poses.z - base_height) * 9.8 * flight._craft._mass / 3600

        if legend: plt.legend(['Solar', 'Propulsion', 'Total'], loc='upper center', bbox_to_anchor=(0.4, 1))

        # Show gravitational potential, if there is any
        if any(E_g > 0):
            EgMax = E_g.max()
            EgMaxRatio = EgMax / bat_Wh_cap
            print('Stored at most %.2f Wh = %.2f%% of battery capacity as gravity potential' % (
                EgMax, EgMaxRatio * 100
            ))


        ax2 = ax1.twinx()
        ax2.xaxis.set_major_formatter(DateFormatter('%H:%M', tz=times.tz))
        ax2.set_ylabel('Stored Energy [Wh]', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        #ax2.set_ylim(bottom = 0, top = (battery + E_g).max() * 1.3)
        ax2.set_ylim(0, 20000)
        #battery.plot(color='tab:red')
        ax2.plot(battery, color='blue', linestyle='-', marker='p', markevery=250, markersize=8)
        if poses['z'].max() - base_height > 10:
            # Show potential energy as well, if we have a reasonable amount
            #(battery + E_g).plot(color='tab:cyan')
            ax2.fill_between(battery.index, battery, battery + E_g, color='blue', alpha=0.2, hatch='/')

        if legend: plt.legend(['Battery', 'Altitude'])

        # Note, we take at least 6 hours to get past the initial charge impact (really only applies to > 24h timespan)
        mSocStart = pd.to_datetime(start) + pd.offsets.Hour(6)
        if mSocStart < pd.to_datetime(end):
            print('mSoc = %.2f%%' % (battery[mSocStart:].min() / bat_Wh_cap * 100))

        # Tidy up x axis margins
        plt.autoscale(enable=True, axis='x', tight=True)

        if savename is not None:
            plt.savefig(savename, bbox_inches = 'tight')

        print()
        print()

    def displayFlightAltitudeThroughputInfo(self, flight, start = '2020-07-01T08', end = '2020-07-03T08'):
        """Displays altitude vs throughput plot."""
        import matplotlib.pyplot as plt
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

        fix, ax1 = plt.subplots()
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

    def runNS3Simulation(self, flight, start = '2020-07-01T08', end = None, burstArrivals = 1, burstLength = 1, upload = True, download = False):
        """
        Runs an NS3 simulation of this flight, one cycle.
        
        Except for short runs, it is better to use the runner*.py files.
        """
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
            
            # For WS paper, upload
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

            # We do a brief look at the download direction too
            for idx in range(len(self._scene.users)):
                # Create traffic generators
                ueNode = sim.ueNodes.Get(idx)
                ueNetDev = sim.ueLteDevs.Get(idx)
                ueIp = sim.ueIpIface.GetAddress(idx)
                
                sinkAddr = ns.network.InetSocketAddress(ueIp, 8000 + idx)
                packetSinkHelper = ns.applications.PacketSinkHelper("ns3::UdpSocketFactory", sinkAddr)
                sim.serverApps.Add(packetSinkHelper.Install(ueNode))
                
                MbpsTarget = 1

                h = ns.applications.PPBPHelper("ns3::UdpSocketFactory", sinkAddr)
                # Supposed to be how many active "bursts" we have, but appears to be how many started per second
                h.SetAttribute("MeanBurstArrivals", StringValue("ns3::ConstantRandomVariable[Constant=%f]" % burstArrivals))
                # Duration in seconds of each burst
                h.SetAttribute("MeanBurstTimeLength", StringValue("ns3::ConstantRandomVariable[Constant=%f]" % burstLength))
                h.SetAttribute("BurstIntensity", StringValue("1Mb/s"))
                app = h.Install(sim.remoteHost)
                sim.clientApps.Add(app)
                
            # Set up traces
            
            
            sim.lteHelper.EnableRlcTraces()
            sim.lteHelper.EnablePhyTraces()
            
            
            flh.InstallAll()
            
            sim.startAndMonitorApps(resolution = 10)
            sim.stopAppsAt(dt)
            
            sim.run(dt + 10)
            
            return sim.serverAppStats, sim.clientAppStats