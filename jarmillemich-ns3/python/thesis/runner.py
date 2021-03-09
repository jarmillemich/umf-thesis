#!/usr/bin/env python
import argparse, random
import pandas as pd
import numpy as np
from matplotlib.pyplot import figure
from thesis.ThesisCraft import craft
from thesis.optimize.functions import (
  SplineyFitnessHelper, batteryReward, throughputReward, throughputPenalty,
  energyPenalty, gravityReward, radiusPenalty, altitudePenalty, energyPenalty
)
from thesis.EvalHelper import Judge
from thesis.Scenario import Scenario

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

scene.addRandomGroundUsersUniformCircular(5, r = args.radius * 1000)
judge = Judge(scene, craft)


times = pd.date_range(start = '2020-11-28T09', end = '2020-11-29T09', freq='10S', tz='America/Detroit').to_series()

######################################

numParticles = 30
numPoints = 24*4
numWaypoints = numPoints * 4
numCodons = numWaypoints * 6

template = [
  1500, 400, 1000, 0, 5, 5,
  1500.5, -400.5, 1000, pi, 5, 5,
  -1500, 400, 1000, pi, 5, 5,
  -1500.5, -400.5, 1000, 0,  5, 5,
] * numPoints
offsets = [50, 50, -5, 0.05, 0.25, 0.25]

#zOffsets = np.random.uniform(-20, 20, numPoints)
zOffsets = np.full(numWaypoints, 0.0)
# Just things (bias initial particles up then down)
gain = 6000
perStep = gain / numWaypoints * 2
print('going up', perStep)
zOffsets[:numPoints*4 // 2] += perStep
zOffsets[numPoints*4 // 2:] -= perStep
template[2::6] = zOffsets.cumsum() + 1100

def createParticle(i):
    if i == 0:
      return template
  
    # This is kinda silly
    np.random.seed(random.randint(0,99999999))
    at = template.copy()
    for i in range(len(template)):
      off = offsets[i % len(offsets)]
      at[i] += np.random.uniform(off / 2, off * 2)
    return at

helper = SplineyFitnessHelper(judge, craft, times, expr = [
  #batteryReward(),
  #gravityReward() * 0.5L,
  radiusPenalty(2000) * 1e-6,
  altitudePenalty(1000, 10000),
  throughputReward() / 1e6 / len(scene.users),
  energyPenalty(0, gravityCoeff = 0)
], desiredDuration = 24*3600)
  
swarm = PSO(numParticles, numCodons, createParticle, helper.getFitness(), processes=30, wSchedule = lambda x: 0.99)

recorder = open('test.rec.txt', 'w')
def writeLn(data):
  recorder.write(str(data) + '\n')
def flush():
  recorder.flush()

def onIteration(i):
  # Save top n vectors
  swarm.snapshot(recorder, args.keep)
  recorder.flush()
    

swarm.iterateMany(iterations = tqdm(range(10)), cb=onIteration)
swarm.dump(recorder)
recorder.close()