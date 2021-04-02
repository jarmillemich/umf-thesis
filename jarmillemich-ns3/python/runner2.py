#!/usr/bin/env python
import os
import argparse, random
import pandas as pd
import numpy as np
from thesis.Aircraft import Aircraft
from thesis.Flight import Flight
from thesis.EvalHelper import Judge
from thesis.Scenario import Scenario
# Our modelled aircraft
from thesis.ThesisCraft import craft

def makeFlight(craft, pathToVec, **kwargs):
  f = open(pathToVec, 'r')
  vec = f.read()
  f.close()
  vec = list(map(float, ''.join(vec.split('\n')).split(',')))

  import argparse, random
  from math import pi
  import pandas as pd
  import numpy as np
  from thesis.ThesisCraft import craft
  from thesis.optimize.functions import (
    SplineyFitnessHelper, Defunc
  )
  from thesis.EvalHelper import Judge
  from thesis.Scenario import Scenario
  from tqdm.auto import tqdm
  from thesis.Flight import Flight

  import random
  random.seed(0)
  np.random.seed(random.randint(0,99999999))

  scene = Scenario()
  scene.addRandomGroundUsersUniformCircular(5, r = 5 * 1000)
  judge = Judge(scene, craft)

  times = pd.date_range(start = '2020-11-28T09', end = '2020-11-29T09', freq='10S', tz='America/Detroit').to_series()


  helper = SplineyFitnessHelper(
    judge, craft, times,
    expr = [
      # Unimportant here
      Defunc(1)
    ],
    # Scale trajectory to evenly fit into a 24-hour window
    desiredDuration = 24*3600,
    # Use our z-scheduling model
    # NB this performs MUCH better than letting the optimizer pick every Z coordinate, or every Z delta
    #    we think this is mostly because these require multi-dimensional coordination, or have global impacts (respectively)
    #    TODO fix this up so the optimizer can pick Z offsets, to do clever local things like angling more towards the sun
    zMode = 'schedule'
  )

  vecToTraj = helper.getTrajBuilder()
  traj, alphas = vecToTraj(vec)

  return Flight(craft, traj, alphas, **kwargs)

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
parser.set_defaults(upload=False)
parser.add_argument('--download', dest='download', action='store_true')
parser.add_argument('--no-download', dest='download', action='store_false')
parser.set_defaults(download=True)


args = parser.parse_args()

print(args)


actualTrajectoryPath = 'optimized-flights/%s.txt' % args.trajectory
if not os.path.isfile(actualTrajectoryPath):
  raise LookupError('Invalid trajectory %s' % args.trajectory)

flight = makeFlight(craft, actualTrajectoryPath, **radioParams)

# We want a fixed scenario over all of our runs, so do a constant seed here
random.seed(0)
np.random.seed(random.randint(0,99999999))

scene = Scenario()
scene.addRandomGroundUsersUniformCircular(5, r = args.radius * 1000)
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
  #end = pd.to_datetime('2020-11-29T08:10'),
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