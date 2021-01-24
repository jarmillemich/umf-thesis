#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"
#include "ns3/mobility-module.h"
#include "ns3/config-store-module.h"
#include "ns3/lte-module.h"
#include "fixed-wing-waycircle-model.h"
// #include "ns3/gtk-config-store.h"
#include <ns3/flow-monitor-helper.h>


using namespace ns3;

/**
 * Sample simulation script for LTE+EPC with different backhauls.
 *
 * The purpose of this example is to compare:
 *
 *  (1) how the simulation user can use a pre-existing EpcHelper that builds
 *      a predefined backhaul network (e.g. the PointToPointEpcHelper) and
 *
 *  (2) how the simulation user can build its custom backhaul network in
 *      the simulation program (i.e. the point-to-point links are created
 *      in the simulation program instead of the pre-existing PointToPointEpcHelper)
 *
 * The pre-existing PointToPointEpcHelper is used with option --useHelper=1 and
 * the custom backhaul is built with option --useHelper=0
 */

NS_LOG_COMPONENT_DEFINE ("FixedWingLteTest");

using namespace std;

/** Helper to figure out what something is made of */
void dumpObjectAggregate(Ptr<const Object> obj) {
  auto it = obj->GetAggregateIterator();
  
  while (it.HasNext()) {
    Ptr<const Object> next = it.Next();
    cout << "has " << next->GetInstanceTypeId().GetName() << endl;
  }
}

struct OurNetwork {
  Ptr<LteHelper> lteHelper;
  Ptr<EpcHelper> epcHelper;
  Ptr<Node> pgw;

  NodeContainer remoteHostContainer;
  Ptr<Node> remoteHost;
  Ipv4Address remoteHostAddr;
  InternetStackHelper internet;

  PointToPointHelper p2ph;

  Ipv4StaticRoutingHelper ipv4RoutingHelper;
  Ptr<Node> sgw;
};

void useOurNetwork(OurNetwork &net) {
  net.lteHelper = CreateObject<LteHelper>();
  net.epcHelper = CreateObject<NoBackhaulEpcHelper>();
  net.lteHelper->SetEpcHelper(net.epcHelper);

  net.pgw = net.epcHelper->GetPgwNode();

  // Create our internet link
  net.remoteHostContainer.Create(1);
  net.remoteHost = net.remoteHostContainer.Get(0);
  net.internet.Install(net.remoteHostContainer);

  // Create the internet (pretending our backhaul is like they hyped up version of Starlink)
  net.p2ph.SetDeviceAttribute ("DataRate", DataRateValue (DataRate ("1Gb/s")));
  net.p2ph.SetDeviceAttribute ("Mtu", UintegerValue (1500));
  net.p2ph.SetChannelAttribute ("Delay", TimeValue (MilliSeconds (20)));
  NetDeviceContainer internetDevices = net.p2ph.Install (net.pgw, net.remoteHost);
  Ipv4AddressHelper ipv4h;
  ipv4h.SetBase ("1.0.0.0", "255.0.0.0");
  Ipv4InterfaceContainer internetIpIfaces = ipv4h.Assign (internetDevices);
  // interface 0 is localhost, 1 is the p2p device
  net.remoteHostAddr = internetIpIfaces.GetAddress (1);

  Ptr<Ipv4StaticRouting> remoteHostStaticRouting = net.ipv4RoutingHelper.GetStaticRouting (net.remoteHost->GetObject<Ipv4> ());
  remoteHostStaticRouting->AddNetworkRouteTo (Ipv4Address ("7.0.0.0"), Ipv4Mask ("255.0.0.0"), 1);

  // SGW Node
  net.sgw = net.epcHelper->GetSgwNode();

  // Install Mobility Model for SGW
  // TODO does position matter here??
  Ptr<ListPositionAllocator> positionAlloc2 = CreateObject<ListPositionAllocator> ();
  positionAlloc2->Add (Vector (0.0,  50.0, 0.0));
  MobilityHelper mobility2;
  mobility2.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility2.SetPositionAllocator (positionAlloc2);
  mobility2.Install (net.sgw);
}

static long totalBytes = 0;

void onPacket(std::string context, Ptr<const Packet> pkt, const Address& addr) {
  // Just keep track of payload size
  totalBytes += pkt->GetSize();
  
  // return;
  
  // auto addrRep = InetSocketAddress::ConvertFrom(addr).GetIpv4();
  
  // std::cout << Simulator::Now().GetMilliSeconds()
  // << " (" << context << ") "
  // << "Packet " << pkt->GetSize()
  // << " from " << addrRep << endl;
}

double lastSinr = 0;

void onSinr(string context, uint16_t cellId, uint16_t rnti, double rsrp, double avSinr, uint8_t componentCarrierId) {
  double sinrDb = 10 * log10(avSinr);
  
  // cout << Simulator::Now().GetMilliSeconds()
  // //<< " (" << context << ")"
  // << " reporting sinr of " << sinrDb
  // << " betwen cell " << cellId
  // << " and rnti " << rnti
  // << endl;

  lastSinr = sinrDb;
}

NodeContainer ueNodes;
NodeContainer enbNodes;

void timeMon() {
  double rate = 8 * (1.0 * totalBytes / 5) / 1024 / 1024;
  
  cout 
    << "T=" << Simulator::Now().GetSeconds()
    << " SINR=" << lastSinr
    << " THRU=" << rate << " Mbps"
    << " dist=" << enbNodes.Get(0)->GetObject<MobilityModel>()->GetPosition()
    << endl;

  auto node = ueNodes.Get(0)->GetDevice(0);
  dumpObjectAggregate(node);

  Simulator::Schedule(Seconds(5), timeMon);
  
  totalBytes = 0;
  lastSinr = 0;
}

int
main (int argc, char *argv[])
{
  // PathMobilityModel foo;
  // foo.AddSegment(CreateObject<PathMobilityModelSegments::LineSegment>(
  //   Vector(0, 0, 0),
  //   Vector(10, 0, 0),
  //   1
  // ));

  // std::cout << foo.GetPosition() << std::endl;
  // return 0;


  Time simTime = Seconds(10.1);
  Time interPacketInterval = MicroSeconds (2000);
  bool disableDl = false;
  bool disableUl = false;

  // TODO hook up traffic generator to Poisson distribution
  // Data to track: delay, loss rate, SINR, throughput
  // TODO probably have energy model in NS3
  // TODO consistency checks for mobility model

  ConfigStore inputConfig;
  inputConfig.ConfigureDefaults ();

  Config::SetDefault("ns3::LteUePhy::RsrpSinrSamplePeriod", UintegerValue(1024));

  OurNetwork net;
  useOurNetwork(net);

  // Install Mobility Model for eNBs and UEs
  Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator> ();
  Ptr<FixedWingWaycircleModel> enbMobility = CreateObject<FixedWingWaycircleModel>();

  /************* CODE PASTING HERE ***************/
positionAlloc->Add(Vector(0,0,0));//976
positionAlloc->Add(Vector(100,0,0));//976
// positionAlloc->Add(Vector(-160.00, 296.00, 0.00));
// positionAlloc->Add(Vector(33.00, 405.00, 0.00));
// positionAlloc->Add(Vector(-313.00, 37.00, 0.00));
// positionAlloc->Add(Vector(217.00, 129.00, 0.00));
// positionAlloc->Add(Vector(410.00, -43.00, 0.00));
// positionAlloc->Add(Vector(95.00, -419.00, 0.00));
// positionAlloc->Add(Vector(403.00, -223.00, 0.00));
// positionAlloc->Add(Vector(291.00, -346.00, 0.00));
// positionAlloc->Add(Vector(-347.00, -300.00, 0.00));
// positionAlloc->Add(Vector(6.00, 411.00, 0.00));
// positionAlloc->Add(Vector(273.00, 52.00, 0.00));
// positionAlloc->Add(Vector(-3.00, -168.00, 0.00));
// positionAlloc->Add(Vector(-85.00, -78.00, 0.00));
// positionAlloc->Add(Vector(360.00, -277.00, 0.00));
// positionAlloc->Add(Vector(-75.00, 164.00, 0.00));
// positionAlloc->Add(Vector(419.00, 133.00, 0.00));
// positionAlloc->Add(Vector(150.00, -273.00, 0.00));
// positionAlloc->Add(Vector(175.00, -452.00, 0.00));
// positionAlloc->Add(Vector(164.00, 79.00, 0.00));
// enbMobility->addLineSegment(Vector(438.80, -50.66, 170.70), Vector(40.38, 40.38, 153.12), 12.56);
// enbMobility->addArcSegment(Vector(-496.09, -50.78, 153.12), 91.60, 1.47, 2.83, 12.04);
// enbMobility->addLineSegment(Vector(-532.42, -134.87, 153.12), Vector(-525.03, -525.03, 124.22), 11.62);
// enbMobility->addArcSegment(Vector(410.16, -433.59, 124.22), 99.61, -1.98, 2.92, 11.09);
// enbMobility->addLineSegment(Vector(468.82, -353.09, 124.22), Vector(32.40, 32.40, 160.55), 11.26);
// enbMobility->addArcSegment(Vector(-109.38, -35.16, 160.55), 83.59, 0.94, 1.69, 10.05);
// enbMobility->addLineSegment(Vector(-182.12, 6.03, 160.55), Vector(-277.53, -277.53, 194.92), 12.29);
// enbMobility->addArcSegment(Vector(-425.78, -230.47, 194.92), 95.51, -0.52, -3.47, 10.55);
// enbMobility->addLineSegment(Vector(-489.25, -159.10, 194.92), Vector(497.78, 497.78, 180.86), 12.60);
// enbMobility->addArcSegment(Vector(289.06, 453.12, 180.86), 59.77, 2.30, -1.94, 10.76);
// enbMobility->addLineSegment(Vector(345.04, 474.05, 180.86), Vector(38.95, 38.95, 170.70), 12.44);
// enbMobility->addArcSegment(Vector(445.31, 15.62, 170.70), 66.60, 0.36, -2.03, 15.59);

enbMobility->addLineSegment(Vector(0, 0, 0), Vector(10000, 0, 0), 10);

  /************ /CODE PASTING HERE ***************/

  
  enbNodes.Create (1);
  Ptr<Node> enb = enbNodes.Get(0);
  ueNodes.Create (positionAlloc->GetSize());

  
  
  MobilityHelper mobility;
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.SetPositionAllocator (positionAlloc);
  mobility.Install (ueNodes);
	enbMobility->finalize();

  //enb->AggregateObject(enbMobility);
  mobility.Install(enbNodes);

  // Install LTE Devices to the nodes
  NetDeviceContainer enbLteDevs = net.lteHelper->InstallEnbDevice (enbNodes);
  NetDeviceContainer ueLteDevs = net.lteHelper->InstallUeDevice (ueNodes);

  Ipv4AddressHelper s1uIpv4AddressHelper;

  // Create networks of the S1 interfaces
  s1uIpv4AddressHelper.SetBase ("10.0.0.0", "255.255.255.252");

  // for (uint16_t i = 0; i < numNodePairs; ++i)
  //   {
  //     Ptr<Node> enb = enbNodes.Get (i);

      // Create a point to point link between the eNB and the SGW with
      // the corresponding new NetDevices on each side
      PointToPointHelper p2ph;
      DataRate s1uLinkDataRate = DataRate ("1Gb/s");
      uint16_t s1uLinkMtu = 20000;
      Time s1uLinkDelay = Time (0);
      p2ph.SetDeviceAttribute ("DataRate", DataRateValue (s1uLinkDataRate));
      p2ph.SetDeviceAttribute ("Mtu", UintegerValue (s1uLinkMtu));
      p2ph.SetChannelAttribute ("Delay", TimeValue (s1uLinkDelay));
      NetDeviceContainer sgwEnbDevices = p2ph.Install (net.sgw, enb);

      Ipv4InterfaceContainer sgwEnbIpIfaces = s1uIpv4AddressHelper.Assign (sgwEnbDevices);
      s1uIpv4AddressHelper.NewNetwork ();

      Ipv4Address sgwS1uAddress = sgwEnbIpIfaces.GetAddress (0);
      Ipv4Address enbS1uAddress = sgwEnbIpIfaces.GetAddress (1);

      // Create S1 interface between the SGW and the eNB
      net.epcHelper->AddS1Interface (enb, enbS1uAddress, sgwS1uAddress);
    //}

  // Install the IP stack on the UEs
  net.internet.Install (ueNodes);
  Ipv4InterfaceContainer ueIpIface;
  ueIpIface = net.epcHelper->AssignUeIpv4Address (NetDeviceContainer (ueLteDevs));
  // Assign IP address to UEs, and install applications
  for (uint32_t u = 0; u < ueNodes.GetN (); ++u)
    {
      Ptr<Node> ueNode = ueNodes.Get (u);
      // Set the default gateway for the UE
      Ptr<Ipv4StaticRouting> ueStaticRouting = net.ipv4RoutingHelper.GetStaticRouting (ueNode->GetObject<Ipv4> ());
      ueStaticRouting->SetDefaultRoute (net.epcHelper->GetUeDefaultGatewayAddress (), 1);
    }

  // Attach one UE per eNodeB
  for (uint16_t i = 0; i < ueLteDevs.GetN(); i++)
    {
      net.lteHelper->Attach (ueLteDevs.Get(i), enbLteDevs.Get(0));
      // side effect: the default EPS bearer will be activated
    }


  // Install and start applications on UEs and remote host
  uint16_t dlPort = 1100;
  uint16_t ulPort = 2000;
  ApplicationContainer clientApps;
  ApplicationContainer serverApps;
  for (uint32_t u = 0; u < ueNodes.GetN (); ++u)
    {
      if (!disableDl)
        {
          PacketSinkHelper dlPacketSinkHelper ("ns3::UdpSocketFactory", InetSocketAddress (Ipv4Address::GetAny (), dlPort));
          serverApps.Add (dlPacketSinkHelper.Install (ueNodes.Get (u)));

          UdpClientHelper dlClient (ueIpIface.GetAddress (u), dlPort);
          dlClient.SetAttribute ("Interval", TimeValue (interPacketInterval));
          dlClient.SetAttribute ("MaxPackets", UintegerValue (100000000));
          clientApps.Add (dlClient.Install (net.remoteHost));

          cout << "Installed download flow (net->UE) from "
               << net.remoteHostAddr
               << " to "
               << ueIpIface.GetAddress (u) << ":" << dlPort
               << endl;
        }

      if (!disableUl)
        {
          ++ulPort;
          PacketSinkHelper ulPacketSinkHelper ("ns3::UdpSocketFactory", InetSocketAddress (Ipv4Address::GetAny (), ulPort));
          serverApps.Add (ulPacketSinkHelper.Install (net.remoteHost));

          UdpClientHelper ulClient (net.remoteHostAddr, ulPort);
          ulClient.SetAttribute ("Interval", TimeValue (interPacketInterval));
          ulClient.SetAttribute ("MaxPackets", UintegerValue (100000000));
          clientApps.Add (ulClient.Install (ueNodes.Get(u)));

          cout << "Installed upload flow (UE->net) from   "
               << ueIpIface.GetAddress (u)
               << " to "
               << net.remoteHostAddr << ":" << ulPort
               << endl;
        }
    }

  serverApps.Start (MilliSeconds (0));
  //clientApps.Start (MilliSeconds (0));
  clientApps.Stop(Seconds(6));
  serverApps.Stop(Seconds(6));
  //net.lteHelper->EnableTraces ();
  net.lteHelper->EnableRlcTraces();
  // Uncomment to enable PCAP tracing
  //net.p2ph.EnablePcapAll("lte-test");

  // Hmm
  std::ostringstream oss;
  // oss << "/NodeList/"
  //     << ueNodes.Get(0)->GetId()
  //     << "/ApplicationList/0"
  //     << "/$ns3::PacketSink/Rx";
  oss << "/NodeList/*/ApplicationList/*/$ns3::PacketSink/Rx";


  Config::Connect(oss.str(), MakeCallback(&onPacket));

  oss.str("");
  oss << "/NodeList/*"
  << "/DeviceList/*"
  << "/ComponentCarrierMapUe/*"
  << "/LteUePhy/ReportCurrentCellRsrpSinr";

  Config::Connect(oss.str(), MakeCallback(&onSinr));

  timeMon();

  NodeContainer allLteNodes;
  allLteNodes.Add(enbNodes);
  allLteNodes.Add(ueNodes);

  FlowMonitorHelper fl;
  Ptr<FlowMonitor> monitor = fl.InstallAll();//Install(allLteNodes);
  
  monitor->SetAttribute("DelayBinWidth", DoubleValue(0.001));
  monitor->SetAttribute("JitterBinWidth", DoubleValue(0.001));
  monitor->SetAttribute("PacketSizeBinWidth", DoubleValue(20));
  
  auto start = std::chrono::high_resolution_clock::now();
  


  Simulator::Stop (simTime);
  Simulator::Run ();

  auto finish = std::chrono::high_resolution_clock::now();
  auto delta = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();
  cout << "That took like " << delta << " ms" << endl;

  monitor->CheckForLostPackets();
  monitor->SerializeToXmlFile("flowResults.xml", true, true);

  /*GtkConfigStore config;
  config.ConfigureAttributes();*/

  Simulator::Destroy ();

  NS_LOG_UNCOND("I guess it worked?");
  oss.str("");
  cout << "Total payload bytes received: " << totalBytes << " = " << (totalBytes >> 20) * 8 << " Mb" << endl;
  cout << "That's like " << ((totalBytes >> 20) * 8 / simTime.GetSeconds()) << " Mbps!" << endl;
  cout << "With our guesstimated overhead, we get " << ((totalBytes >> 20) * 8 / simTime.GetSeconds() / 0.9333) << " Mbps" << endl;
  NS_LOG_UNCOND(oss.str());  

  cout << "and now, the weather! (" << JM_timing.size() << ")" << endl;

  

  for (auto it : JM_timing) {
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(it.second).count();
    if (ms > 1000) {
      std::cout << it.first << ": " << ms << " over " << JM_count[it.first] << std::endl;
    }
  }

  // for (auto it : JM_count) {
  //   std::cout << it.first << ": " << it.second << std::endl;
  // }

  return 0;
}
