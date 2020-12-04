import ns.core
import ns.applications
import ns.internet
import ns.network
import ns.lte
import ns.point_to_point
import ns.mobility

from ns.core import Vector, StringValue, DoubleValue, TimeValue, UintegerValue, MicroSeconds, MilliSeconds, Seconds
from ns.internet import Ipv4
from ns.network import Ipv4Address, Ipv4Mask

# Mostly adapted from lte/examples/lena-simple-epc-backhaul
class SimulationContext():
    def __init__(self):
        # This is just a place to chuck references that we don't want
        # to get garbage collected because they fall out of scope
        self.refs = []
        
        # Stats
        self.serverAppStats = []
        self.serverAppsLastSample = []
        
    def _ref(self, item):
        self.refs.append(item)
    
    def __enter__(self):
        # Reset our IP address space when we re-setup the simulation
        # XXX This may be ill-advised...
        ns.internet.Ipv4AddressGenerator.Reset()

        # TODO move a bunch of these we don't care about into _ref instead of naming
        
        # NB These MUST NOT be re-invoked in the same process without resetting IP address space
        self.lteHelper = ns.lte.LteHelper()
        self.epcHelper = ns.lte.NoBackhaulEpcHelper()
        self.ipv4h = ns.internet.Ipv4AddressHelper()
        
        self.lteHelper.SetEpcHelper(self.epcHelper)
        self.pgw = self.epcHelper.GetPgwNode()
        
        # Create our remote host and internet link
        self.remoteHostContainer = ns.network.NodeContainer()
        self.remoteHostContainer.Create(1)
        self.remoteHost = self.remoteHostContainer.Get(0)
        self.internet = ns.internet.InternetStackHelper()
        self.internet.Install(self.remoteHostContainer)
        
        # Create our backhaul
        self.p2ph = ns.point_to_point.PointToPointHelper()

        self.p2ph.SetDeviceAttribute("DataRate", StringValue("1Gbps"))
        self.p2ph.SetDeviceAttribute("Mtu", UintegerValue(int(1500)))
        self.p2ph.SetChannelAttribute("Delay", TimeValue(MilliSeconds(int(20))))

        self.internetDevices = self.p2ph.Install(self.pgw, self.remoteHost)

        self.ipv4h.SetBase(Ipv4Address("1.0.0.0"), Ipv4Mask("255.0.0.0"))
        self.internetIpInterfaces = self.ipv4h.Assign(self.internetDevices)
        # 0 is loopback, 1 is p2p
        self.remoteHostAddr = self.internetIpInterfaces.GetAddress(1)

        self.routingHelper = ns.internet.Ipv4StaticRoutingHelper()
        self.remoteHostIpv4 = self.remoteHost.GetObject(Ipv4.GetTypeId())
        self.remoteHostStaticRouting = self.routingHelper.GetStaticRouting(self.remoteHostIpv4)
        self.remoteHostStaticRouting.AddNetworkRouteTo(Ipv4Address("7.0.0.0"), Ipv4Mask("255.0.0.0"), 1)

        # SGW node
        self.sgw = self.epcHelper.GetSgwNode()

        # Apparently SGW needs a mobility model?
        self.sgwLp = ns.mobility.ListPositionAllocator()
        self.sgwLp.Add(ns.core.Vector(0, 0, 0))
        self.sgwMob = ns.mobility.MobilityHelper()
        self.sgwMob.SetMobilityModel("ns3::ConstantPositionMobilityModel")
        self.sgwMob.SetPositionAllocator(self.sgwLp)
        self.sgwMob.Install(self.sgw)
        
        # UE and ENB Nodes, to be populated
        self.ueNodes = ns.network.NodeContainer()
        self.enbNodes = ns.network.NodeContainer()
        
        self.userPositions = ns.mobility.ListPositionAllocator()
        self.enbMobilities = []

        # Application containers
        self.clientApps = ns.network.ApplicationContainer()
        self.serverApps = ns.network.ApplicationContainer()

        self.dlPort = 1100
        self.ulPort = 2000

        return self
        
    def __exit__(self, exception_type, exception_value, exception_traceback):
        # I guess there's nothing to do here since we reset IP space on entry
        ns.core.Simulator.Destroy()
    
    # Create the backhaul to the SGW for the provided ENB node
    def _createBackhaul(self, enbNode, throughput = '1Gbps', delay=0):
        addrHelper = ns.internet.Ipv4AddressHelper()
        # Might not need this one...
        self._ref(addrHelper)
        
        addrHelper.SetBase(Ipv4Address("10.0.0.0"), Ipv4Mask("255.255.255.252"))
        
        p2ph = ns.point_to_point.PointToPointHelper()
        p2ph.SetDeviceAttribute("DataRate", StringValue(throughput))
        p2ph.SetDeviceAttribute("Mtu", UintegerValue(2000))
        p2ph.SetChannelAttribute("Delay", TimeValue(MilliSeconds(delay)))
        sgwEnbDevices = p2ph.Install(self.sgw, enbNode)
        
        sgwEnbIpIfaces = addrHelper.Assign(sgwEnbDevices)
        addrHelper.NewNetwork()
        
        sgwS1uAddress = sgwEnbIpIfaces.GetAddress(0)
        enbS1uAddress = sgwEnbIpIfaces.GetAddress(1)
        
        self.epcHelper.AddS1Interface(enbNode, enbS1uAddress, sgwS1uAddress)
    
    def _finalizeNodes(self):
        # Create ENB nodes and assign mobility
        for mobility in self.enbMobilities:
            idx = self.enbNodes.GetN()
            self.enbNodes.Create(1)
            enb = self.enbNodes.Get(idx)
            
            # TODO we should check if there is already a model?
            enb.AggregateObject(mobility)

        # Create UE nodes
        self.ueNodes.Create(self.userPositions.GetSize())
        ueMob = ns.mobility.MobilityHelper()
        ueMob.SetMobilityModel("ns3::ConstantPositionMobilityModel")
        ueMob.SetPositionAllocator(self.userPositions)
        ueMob.Install(self.ueNodes)
            
        # Install LTE devices
        self.enbLteDevs = self.lteHelper.InstallEnbDevice(self.enbNodes)
        self.ueLteDevs = self.lteHelper.InstallUeDevice(self.ueNodes)
        
        # Create backhauls
        for idx in range(self.enbNodes.GetN()):
            enbNode = self.enbNodes.Get(idx)
            self._createBackhaul(enbNode)
        
        # Install IP stack on UEs
        self.internet.Install(self.ueNodes)
        self.ueIpIface = self.epcHelper.AssignUeIpv4Address(ns.network.NetDeviceContainer(self.ueLteDevs))
        for idx in range(self.ueNodes.GetN()):
            ueNode = self.ueNodes.Get(idx)
            ueStaticRouting = self.routingHelper.GetStaticRouting(ueNode.GetObject(Ipv4.GetTypeId()))
            ueStaticRouting.SetDefaultRoute(self.epcHelper.GetUeDefaultGatewayAddress(), 1)

            # Just attaching to the first ENB for now
            self.lteHelper.Attach(self.ueLteDevs.Get(idx), self.enbLteDevs.Get(0))

    # Creates a UDP traffic generator
    def createTrafficGenerator(self, ueIdx, uploadDir = False, interval = 100):
        ueNode = self.ueNodes.Get(ueIdx)
        ueNetDev = self.ueLteDevs.Get(ueIdx)
        ueIp = self.ueIpIface.GetAddress(ueIdx)

        if uploadDir:
            # Upload (UE -> Internet)
            
            # Sink on the internet host
            self.ulPort += 1
            sinkAddr = ns.network.InetSocketAddress(Ipv4Address.GetAny(), self.ulPort)
            packetSinkHelper = ns.applications.PacketSinkHelper("ns3::UdpSocketFactory", sinkAddr)
            self.serverApps.Add(packetSinkHelper.Install(self.remoteHost))

            # Generator on the UE
            ulClient = ns.applications.UdpClientHelper(self.remoteHostAddr, self.ulPort)
            ulClient.SetAttribute("Interval", TimeValue(MicroSeconds(interval)))
            ulClient.SetAttribute("MaxPackets", UintegerValue(int(2**32-1)))
            ueContainer = ns.network.NodeContainer(ueNode)
            self.clientApps.Add(ulClient.Install(ueContainer))
        else:
            # Download (Internet -> UE)

            # Sink on the UE
            sinkAddr = ns.network.InetSocketAddress(Ipv4Address.GetAny(), self.dlPort)
            packetSinkHelper = ns.applications.PacketSinkHelper("ns3::UdpSocketFactory", sinkAddr)
            self.serverApps.Add(packetSinkHelper.Install(ueNode))

            # Generator on the internet host
            dlClient = ns.applications.UdpClientHelper(ueIp, self.dlPort)
            dlClient.SetAttribute("Interval", TimeValue(MicroSeconds(interval)))
            dlClient.SetAttribute("MaxPackets", UintegerValue(int(2**32-1)))
            self.clientApps.Add(dlClient.Install(self.remoteHostContainer))

            print(ueIp)
            
    def startAndMonitorApps(self, resolution = 5):
        self.serverApps.Stop(Seconds(0))
        self.clientApps.Stop(Seconds(0))
        
        # Set up a stats gathering event
        self.serverAppsLastSample = [0 for i in range(self.serverApps.GetN())]
        self.monitorApps(resolution)
        
    def monitorApps(self, resolution = 5):
        nApps = self.serverApps.GetN()
        at = [self.serverApps.Get(i).GetTotalRx() for i in range(nApps)]
        delta = [at[i] - self.serverAppsLastSample[i] for i in range(nApps)]
        self.serverAppsLastSample = at
        
        self.serverAppStats.append(delta)
        
        #print('at %d got %s' % (ns.core.Simulator.Now().GetSeconds(), delta))
        #print(self.enbNodes.Get(0).GetObject(ns.mobility.MobilityModel.GetTypeId()).GetPosition())
        
        ns.core.Simulator.Schedule(Seconds(float(resolution)), lambda: self.monitorApps(resolution))
        
    def stopAppsAt(self, dt):
        self.serverApps.Stop(Seconds(float(dt)))
        self.clientApps.Stop(Seconds(float(dt)))
    
    def run(self, time):
        from tqdm.notebook import tqdm
        pbar = tqdm(unit = ' sim seconds', total = int(time))
        
        # Keep track of time
        def logTime():
            now = ns.core.Simulator.Now().GetSeconds()
            pbar.n = int(now)
            pbar.update(0)
            ns.core.Simulator.Schedule(Seconds(1), logTime)
            
        logTime()

        # Start apps
        self.serverApps.Start(MilliSeconds(0))
        self.clientApps.Start(MilliSeconds(0))
        
        ns.core.Simulator.Stop(Seconds(time))
        ns.core.Simulator.Run()
        pbar.n = pbar.total
        pbar.refresh()
        
    def addUser(self, x, y, z):
        self.userPositions.Add(Vector(x, y, z))
        
    def addEnbFlight(self, flight):
        model = flight.toSim()
        self.enbMobilities.append(model)