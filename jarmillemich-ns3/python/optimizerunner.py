#!/usr/bin/env python
import argparse, random
from math import pi
import pandas as pd
import numpy as np
from thesis.ThesisCraft import craft
from thesis.optimize.functions import (
  SplineyFitnessHelper, batteryReward, throughputReward, throughputPenalty,
  energyPenalty, gravityReward, radiusPenalty, altitudePenalty, energyPenalty,
  speedPenalty, alphaPenalty, thrustPenalty
)
from thesis.EvalHelper import Judge
from thesis.Scenario import Scenario
from tqdm.auto import tqdm

from thesis.optimize.PSOv2 import PSO
from thesis.optimize.ParallelNelderMead import ParallelNelderMead

# Parse arguments
parser = argparse.ArgumentParser(description = 'Runner for NS3 simulations')
parser.add_argument('--run', type=int, help='Python random seed', default=0)
parser.add_argument('--users', type=int, help='Number of users', default=5)
parser.add_argument('--radius', type=float, help='Radius of user (km)', default=5)
parser.add_argument('--mode', type=str, help='pso, nm', default='pso')
parser.add_argument('--k', type=int, help='For Nelder-Mead, # of vertices to update in parallel', default=128)
parser.add_argument('--threads', type=int, help='# Of threads to use (actually # of processes)', default=4)
parser.add_argument('--iterations', type=int, help='# Of iterations', default=1000)
parser.add_argument('--energy', type=float, help='Delta Energy constraint (Wh, negative is require gain)', default=0)

parser.add_argument('--keep', type=float, help='Keep top N vectors', default=3)
args = parser.parse_args()
print(args)

its = args.iterations
numParticles = args.k # W/E

opath = 'out/stats-v1-PSO-%dp-r%d-%du-%.2fkm-op_%s-%dit-e%d.txt' % (
  numParticles,
  args.run,
  args.users,
  args.radius,
  args.mode,
  its,
  args.energy
)

# Set up random state right away
import random
random.seed(args.run)
np.random.seed(random.randint(0,99999999))

scene = Scenario()
scene.addRandomGroundUsersUniformCircular(5, r = args.radius * 1000)
judge = Judge(scene, craft)

# Use a time pretty far into winter, 24 hours
times = pd.date_range(start = '2020-11-28T09', end = '2020-11-29T09', freq='10S', tz='America/Detroit').to_series()

######################################

# 4 circles per hour, approximately
numPoints = 4 * 24
numWaypoints = numPoints * 4
numCodons = 6 + numWaypoints * 5

bat_cap = 15.5 * 650
charge_start = 0.5*bat_cap
charge_start_ratio = charge_start / bat_cap

# Where we start
# First part is z Schedule (gain, rest_start, ascend, sustain, descend, rest_end)
# Second part is a circle with four control points (evenly spaced)
# Additional 0.1..0.4s are just to account for our bi-arc model being incomplete on edge cases
template = [3000, 0.1, 0.5, 0.5, 0.5, 0.5] + [
    0.4, 1800.1, 0, 5, 5,
    1800.3, 0.2, 3*pi/2, 5, 5,
    0.2, -1800.3, pi, 5, 5,
    -1800.1, 0.4, pi / 2, 5, 5,
] * numPoints
# How much to offset our initial guesses
# Larger numbers maybe implies more impulse
offsets = [200, 0.1, 0.1, 0.1, 0.1, 0.1] + [50, 50, 0.05, 0.5, 0.5] * 4 * numPoints

# Where to stop PSO from shooting out into the void
bounds = [(0, 9000)] + [
  (0.01, 1)
] * 5 + [
  # XYZ
  (-2000, 2000),
  (-2000, 2000),
  #(1000, 10000),
  # heading
  (None, None), # TODO make a proper distance formula for direction
  # Alphas
  (0, 12),
  (0, 12),
] * numWaypoints


# Surprisingly, NM does much better with this than the one that guaranteed a good basis
# TODO figure out what our mean/variance of flatness is with this or analogous case?
def createParticle(i):
  if i == 0:
    return template

  at = template.copy()
  for i in range(len(template)):
    off = offsets[i % len(offsets)]
    at[i] += np.random.uniform(-off * 2, off * 2)
  return at

helper = SplineyFitnessHelper(
  judge, craft, times,
  expr = [
    # === Optimize this ===
    # Throughput, in Mbps/user
    throughputReward() / 1e6 / len(scene.users),

    # === Subject to these constraints (as penalty functions) ===
    # Flight volume constraints
    radiusPenalty(2000) * 1e-6,
    altitudePenalty(1000, 10000),
    # We have seen the best results with a 1.0 gravityCoeff, interestingly enough
    # But we should try tweaking this again later
    energyPenalty(args.energy, gravityCoeff = 1.0),
    # Some aircraft/modelling constraints
    # (importantly keeps the optimizer from discovering "rocket mode", which we don't model correctly)
    thrustPenalty(hi = 100),
    speedPenalty(lo = 6, hi = 25)
  ],
  # Scale trajectory to evenly fit into a 24-hour window
  desiredDuration = 24*3600,
  # Use our z-scheduling model
  # NB this performs MUCH better than letting the optimizer pick every Z coordinate, or every Z delta
  #    we think this is mostly because these require multi-dimensional coordination, or have global impacts (respectively)
  #    TODO fix this up so the optimizer can pick Z offsets, to do clever local things like angling more towards the sun
  zMode = 'schedule'
)

baselineVec = createParticle(1)
print('  baseline fitness is %.2f=%s' % (
  helper.getFitness(initial_charge = charge_start_ratio, debug=False)(baselineVec),
  helper.getFitness(initial_charge = charge_start_ratio, debug=True)(baselineVec)
))


fitness = helper.getFitness(initial_charge = charge_start_ratio)

  
if args.mode == 'pso':
  def wRamp(it):
    # 0.99 gave reasonable results, but had a lot of fast particles still...
    from math import sqrt
    return 0.97 / (1 + sqrt(it) / 500)

  optimizer = PSO(numParticles, numCodons, createParticle, fitness, processes=args.threads, wSchedule = wRamp, bounds=bounds)
elif args.mode == 'nm':
  optimizer = ParallelNelderMead(numCodons, createParticle, fitness, processes=args.threads, k=args.k, adaptive = True)



recorder = open(opath, 'w')
def write(data):
  if type(data) == list:
    data = ','.join([str(round(x, 4)) for x in data])

  recorder.write(str(data) + '\n')
def flush():
  recorder.flush()

iter = tqdm(range(args.iterations))

def onIteration(i):
  # Save top n vectors
  if args.mode == 'pso':
    best = optimizer.particles[-1][2]
    worst = optimizer.particles[0][2]
    bestAll = optimizer.best[1]
    vels = np.array([vw[1].length() for vw in optimizer.particles])
    iter.set_description('PSO %.2f-%.2f/%.2f~%.2f f(x%d)' % (worst, best, bestAll, vels.mean(), fitness.evaluations.value))
  elif args.mode == 'nm':
    best = optimizer.vertices[-1].fitness
    worst = optimizer.vertices[0].fitness
    # Hypothetically, these two vectors COULD be the furthest separated
    # (probably not though, I think the hyperspace we're in is pretty cliffy/bumpy)
    fakevel = (optimizer.vertices[-1].vec - optimizer.vertices[0].vec).length()
    iter.set_description('NM %.2f-%.2f~%.2f f(x%d)' % (worst, best, fakevel, fitness.evaluations.value))
  
  optimizer.snapshot(write, args.keep)
  recorder.flush()
    
try:
  optimizer.iterateMany(iterations = iter, cb=onIteration)
finally:
  optimizer.dump(write)
  recorder.close()