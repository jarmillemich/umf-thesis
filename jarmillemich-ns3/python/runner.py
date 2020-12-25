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
mass, wingSpan, e0, chord, bat_Wh_cap = 51.90, 24.00, 0.92, 0.50, 4000.00

trajectories = {
  'circle': CircleTrajectory((0, 0, 1000), 1995.05),
  'bowtie': BowtieTrajectory((0, 0, 1000), lobeRadius = 500, lobeCenterDistance = 1494.4),
  'ladder': SimpleLadderTrajectory(
    (0, 0, 1000),
    lobeRadius = 500,
    lobeCenterDistance = 1495.6,
    stepHeight=60,
    nSteps=47
  )
}

# Our modelled aircraft
craft = Aircraft(mass = mass, wingSpan = wingSpan, e0 = 0.92, chord = chord)

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

args = parser.parse_args()

print(args)

if args.trajectory not in trajectories:
  raise LookupError('Invalid trajectory %s' % args.trajectory)
trajectory = trajectories[args.trajectory]

flight = Flight(craft, trajectory, [5 for piece in trajectory.pieces], **radioParams)
# If we wanted to have these depend on the run #, change this to args.run instead of a constant
random.seed(0)
scene = Scenario()
scene.addRandomGroundUsersUniformCircular(1, r = 5000)

judge = Judge(scene, craft)
print(scene.users)
judge.displayFlightTrajectoryInfo(flight, render = False)

from ns.core import RngSeedManager, StringValue, Config
RngSeedManager.SetRun(args.run)

Config.SetDefault("ns3::RadioBearerStatsCalculator::DlRlcOutputFilename", StringValue("out/DlRlcStats.txt"))
Config.SetDefault("ns3::RadioBearerStatsCalculator::UlRlcOutputFilename", StringValue("out/UlRlcStats.txt"))
Config.SetDefault("ns3::RadioBearerStatsCalculator::DlPdcpOutputFilename", StringValue("out/DlPdcpStats.txt"))
Config.SetDefault("ns3::RadioBearerStatsCalculator::UlPdcpOutputFilename", StringValue("out/UlPdcpStats.txt"))

Config.SetDefault("ns3::PhyStatsCalculator::DlRsrpSinrFilename", StringValue("out/DlRsrpSinrStats.txt"))
Config.SetDefault("ns3::PhyStatsCalculator::UlSinrFilename", StringValue("out/UlSinrStats.txt"))
Config.SetDefault("ns3::PhyStatsCalculator::UlInterferenceFilename", StringValue("out/UlInterferenceStats.txt"))

Config.SetDefault("ns3::PhyRxStatsCalculator::DlRxOutputFilename", StringValue("out/DlRxPhyStats.txt"))
Config.SetDefault("ns3::PhyRxStatsCalculator::UlRxOutputFilename", StringValue("out/UlRxPhyStats.txt"))
Config.SetDefault("ns3::PhyTxStatsCalculator::DlTxOutputFilename", StringValue("out/DlTxPhyStats.txt"))
Config.SetDefault("ns3::PhyTxStatsCalculator::UlTxOutputFilename", StringValue("out/UlTxPhyStats.txt"))

serverStats, clientStats = judge.runNS3Simulation(
  flight,
  # One 24 hour period
  start = '2020-07-01T08',
  end = '2020-07-02T08'
)

# Convert index to T+{x} minutes
# 10 is the sampling interval in simulation.py
index = [float(i * 10 / 60) for i in range(len(serverStats) - 1)]
pd.DataFrame(serverStats[1:], index=index).to_csv('./out/%s_%s_server.csv' % (args.trajectory, args.run))
pd.DataFrame(clientStats[1:], index=index).to_csv('./out/%s_%s_client.csv' % (args.trajectory, args.run))