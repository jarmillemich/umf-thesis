#!/usr/bin/env python
import argparse, random
from math import pi
import pandas as pd
import numpy as np
from matplotlib.pyplot import figure
from thesis.ThesisCraft import craft
from thesis.optimize.functions import (
  SplineyFitnessHelper, batteryReward, throughputReward, throughputPenalty,
  energyPenalty, gravityReward, radiusPenalty, altitudePenalty, energyPenalty,
  speedPenalty, alphaPenalty
)
from thesis.EvalHelper import Judge
from thesis.Scenario import Scenario
from tqdm.auto import tqdm

from thesis.optimize.PSOv2 import PSO

# Parse arguments
parser = argparse.ArgumentParser(description = 'Runner for NS3 simulations')
parser.add_argument('--trajectory', metavar='trajectory', type=str, help='circle, bowtie, ladder', default='circle')
parser.add_argument('--run', type=int, help='Python random seed', default=0)
parser.add_argument('--users', type=int, help='Number of users', default=5)
parser.add_argument('--radius', type=float, help='Radius of user (km)', default=5)
parser.add_argument('--keep', type=float, help='Keep top N vectors', default=3)
args = parser.parse_args()
print(args)


scene = Scenario()
import random

random.seed(args.run)
np.random.seed(random.randint(0,99999999))

scene.addRandomGroundUsersUniformCircular(5, r = args.radius * 1000)
judge = Judge(scene, craft)


times = pd.date_range(start = '2020-11-28T09', end = '2020-11-29T09', freq='10S', tz='America/Detroit').to_series()

######################################

numParticles = 300
numPoints = 24*4
numWaypoints = numPoints * 4
numCodons = numWaypoints * 6

# template = [
#   1500, 400, 1000, 0, 5, 5,
#   1500.5, -400.5, 1000, pi, 5, 5,
#   -1500, 400, 1000, pi, 5, 5,
#   -1500.5, -400.5, 1000, 0,  5, 5,
# ] * numPoints
# offsets = [50, 50, -5, 0.05, 0.25, 0.25]

# #zOffsets = np.random.uniform(-20, 20, numPoints)
# zOffsets = np.full(numWaypoints, 0.0)
# # Just things (bias initial particles up then down)
# gain = 6000
# perStep = gain / numWaypoints * 2
# print('going up', perStep)
# zOffsets[:numPoints*4 // 2] += perStep
# zOffsets[numPoints*4 // 2:] -= perStep
# template[2::6] = zOffsets.cumsum() + 1100

template = []

rOutr = 479.9
rCntr = 1335
dt=0.2675

stepDownHeight=70
nSteps=41
nStepsDown=70
stepHeight = stepDownHeight * nStepsDown / nSteps

for i in range(2*nSteps):
  if i % 2 == 0:
    template.extend([
      -rCntr+0.01, -rOutr, 1000 + i * stepHeight, dt, 5, 5,
      rCntr, rOutr, 1000 + (i + 1) * stepHeight, dt, 5, 5,
    ])
  else:
    template.extend([
      rCntr+0.01, -rOutr, 1000 + i * stepHeight, pi-dt, 5, 5,
      -rCntr, rOutr, 1000 + (i + 1) * stepHeight, pi-dt, 5, 5,
    ])
    
altMax = 2 * nSteps * stepHeight + 1000
    
for i in range(2*nStepsDown):
  if i % 2 == 0:
    template.extend([
      -rCntr+0.01, -rOutr, altMax - i * stepDownHeight, dt, 5, 5,
      rCntr, rOutr, altMax - (i+1) * stepDownHeight, dt, 5, 5,
    ])
  else:
    template.extend([
      rCntr+0.01, -rOutr, altMax - i * stepDownHeight, pi-dt, 5, 5,
      -rCntr, rOutr, altMax - (i+1) * stepDownHeight, pi-dt, 5, 5,
    ])

offsets = [50, 50, -5, 0.05, 0.25, 0.25]


def createParticle(i):
    if i == 0:
      return template
  
    at = template.copy()
    for i in range(len(template)):
      off = offsets[i % len(offsets)]
      at[i] += np.random.uniform(off / 2, off * 5)
    return at

helper = SplineyFitnessHelper(judge, craft, times, expr = [
  throughputReward() / 1e6 / len(scene.users),
  #batteryReward(),
  #gravityReward() * 0.5L,
  # Stay inside our region
  radiusPenalty(2000) * 1e-6,
  altitudePenalty(900, 12000),
  # Don't lose energy
  energyPenalty(0, gravityCoeff = 0),
  # Keep inside the domain where our model works
  alphaPenalty(), speedPenalty(7, 25)
], desiredDuration = 24*3600)
  
swarm = PSO(
  numParticles, len(template),
  createParticle, helper.getFitness(),
  processes=10, wSchedule = lambda x: 0.95
)

opath = 'out/stats-PSO-%dp-r%d-%du-%.2fkm.txt' % (
  numParticles,
  args.run,
  args.users,
  args.radius
)

recorder = open(opath, 'w')
def write(data):
  if type(data) == list:
    data = ','.join([str(round(x, 4)) for x in data])

  recorder.write(str(data) + '\n')
def flush():
  recorder.flush()

iter = tqdm(range(5000))

def onIteration(i):
  # Save top n vectors
  best = swarm.particles[-1][2]
  worst = swarm.particles[0][2]
  bestAll = swarm.best[1]
  vels = np.array([vw[1].length() for vw in swarm.particles])
  iter.set_description('Beep %.2f-%.2f/%.2f~%.2f' % (worst, best, bestAll, vels.mean()))
  swarm.snapshot(write, args.keep)
  recorder.flush()

  #print(helper.getFitness(debug=True)(swarm.best[0]))
    
try:
  swarm.iterateMany(iterations = iter, cb=onIteration)
finally:
  swarm.dump(write)
  recorder.close()