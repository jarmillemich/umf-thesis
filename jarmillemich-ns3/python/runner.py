#!/usr/bin/env python
import argparse, random
import pandas as pd
import numpy as np
from thesis.Aircraft import Aircraft
from thesis.Flight import Flight
from thesis.EvalHelper import Judge
from thesis.Scenario import Scenario


from thesis.Trajectory import CircleTrajectory, BowtieTrajectory, SimpleLadderTrajectory

# These values are derived as described in the Trial 3 notebook
mass, wingSpan, e0, chord = 85.8526, 21.0000, 0.92, 1.0244

trajectories = {
  'circle': CircleTrajectory((0, 0, 1000), 1972.03),
  'bowtie': BowtieTrajectory((0, 0, 1000), lobeRadius = 500, lobeCenterDistance = 1476.23),
  'ladder': SimpleLadderTrajectory(
    (0, 0, 1000),
    lobeRadius = 500,
    lobeCenterDistance = 1475.45,
    stepHeight=60,
    nSteps=36,
    nStepsDown=55
  )
}

# Our modelled aircraft
craft = Aircraft(mass = mass, wingSpan = wingSpan, e0 = e0, chord = chord)

# From NS3 defaults
radioParams = {
  'xmitPower': 30, # dBm
  'B': 180e3 * 25, # 25 180kHz RBs = 4.5 MHz
  'N0': -174       # See lte-spectrum-value-helper.cc kT_dBm_Hz
}


# Parse arguments
parser = argparse.ArgumentParser(description = 'Runner for NS3 simulations')
parser.add_argument('--trajectory', metavar='trajectory', type=str, help='circle, bowtie, ladder, dleft, dright', default='circle')
parser.add_argument('--run', type=int, help='NS3 RNG run number', default=0)
parser.add_argument('--users', type=int, help='Number of users', default=5)
parser.add_argument('--radius', type=float, help='Radius of user (km)', default=5)

parser.add_argument('--upload', dest='upload', action='store_true')
parser.add_argument('--no-upload', dest='upload', action='store_false')
parser.set_defaults(upload=True)
parser.add_argument('--download', dest='download', action='store_true')
parser.add_argument('--no-download', dest='download', action='store_false')
parser.set_defaults(download=False)


args = parser.parse_args()

print(args)

if args.trajectory not in trajectories:
  raise LookupError('Invalid trajectory %s' % args.trajectory)
trajectory = trajectories[args.trajectory]

flight = Flight(craft, trajectory, [5 for piece in trajectory.pieces], **radioParams)
# If we wanted to have these depend on the run #, change this to args.run instead of a constant
random.seed(args.users)
scene = Scenario()
scene.addRandomGroundUsersUniformCircular(args.users, r = args.radius * 1000)

judge = Judge(scene, craft)
print(scene.users)
judge.displayFlightTrajectoryInfo(flight, render = False)

from ns.core import RngSeedManager, StringValue, UintegerValue, Config
import ns.lte # Only needed so attributes are registered properly
RngSeedManager.SetRun(args.run)

def mkPathRaw(name):
  return './out/%s_%s_run%d_usr%d_rad%.2f_ul%d_dl%d.txt' % (name, args.trajectory, args.run, args.users, args.radius, args.upload, args.download)

def mkPath(name):
  return StringValue(mkPathRaw(name))

Config.SetDefault("ns3::RadioBearerStatsCalculator::DlRlcOutputFilename", mkPath("DlRlcStats"))
Config.SetDefault("ns3::RadioBearerStatsCalculator::UlRlcOutputFilename", mkPath("UlRlcStats"))
Config.SetDefault("ns3::RadioBearerStatsCalculator::DlPdcpOutputFilename", mkPath("DlPdcpStats"))
Config.SetDefault("ns3::RadioBearerStatsCalculator::UlPdcpOutputFilename", mkPath("UlPdcpStats"))

Config.SetDefault("ns3::PhyStatsCalculator::DlRsrpSinrFilename", mkPath("DlRsrpSinrStats"))
Config.SetDefault("ns3::PhyStatsCalculator::UlSinrFilename", mkPath("UlSinrStats"))

# Heavy and not helpful, for now
Config.SetDefault("ns3::PhyStatsCalculator::UlInterferenceFilename", StringValue("/dev/null"))
Config.SetDefault("ns3::PhyRxStatsCalculator::DlRxOutputFilename", StringValue("/dev/null"))
Config.SetDefault("ns3::PhyRxStatsCalculator::UlRxOutputFilename", StringValue("/dev/null"))
Config.SetDefault("ns3::PhyTxStatsCalculator::DlTxOutputFilename", StringValue("/dev/null"))
Config.SetDefault("ns3::PhyTxStatsCalculator::UlTxOutputFilename", StringValue("/dev/null"))

# Approximately the ms sampling interval for SINR/RSRP
Config.SetDefault("ns3::LteEnbPhy::UeSinrSamplePeriod", UintegerValue(100))
Config.SetDefault("ns3::LteUePhy::RsrpSinrSamplePeriod", UintegerValue(100))


serverStats, clientStats = judge.runNS3Simulation(
  flight,
  # One 24 hour period
  start = pd.to_datetime('2020-11-29T08'),
  # Test time
  #end = pd.to_datetime('2020-11-29T09'),
  # WS time
  end = pd.to_datetime('2020-11-30T08'),
  upload=args.upload,
  download=args.download
)

# Convert index to T+{x} minutes
# 10 is the sampling interval in simulation.py
index = [float(i * 10 / 60) for i in range(len(serverStats) - 1)]
pd.DataFrame(serverStats[1:], index=index).to_csv(mkPathRaw('serverStats'))
pd.DataFrame(clientStats[1:], index=index).to_csv(mkPathRaw('clientStats'))