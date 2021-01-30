from math import sqrt
from multiprocessing import Pool
from queue import Queue
import random

class Vector:
  def __init__(self, pos):
    self.pos = pos

  def __add__(self, other):
    if len(self.pos) != len(other.pos):
      raise TypeError('Vectors must be of the same length to __add__')

    newPos = self.pos.copy()
    for i in range(len(self.pos)):
      newPos[i] += other.pos[i]

    return Vector(newPos)

  def __sub__(self, other):
    if len(self.pos) != len(other.pos):
      raise TypeError('Vectors must be of the same length to __sub__')

    newPos = self.pos.copy()
    for i in range(len(self.pos)):
      newPos[i] -= other.pos[i]
    
    return Vector(newPos)

  def __mul__(self, scalar):
    newPos = self.pos.copy()
    for i in range(len(self.pos)):
      newPos[i] *= scalar
    
    return Vector(newPos)

  def __rmul__(self, scalar):
    return self * scalar

  def __div__(self, scalar):
    return self * (1 / scalar)

  def __repr__(self):
    return 'Vector<' + ','.join([str(round(p,2)) for p in self.pos]) + '>'

  def length(self):
    sum = 0
    for p in self.pos:
      sum += p ** 2
    return sqrt(sum)

  def perturb(self):
    r = random.random()
    # Comment/uncommet for element vs whole vector multiplication
    #self.pos = [p * r for p in self.pos]self.pos
    self.pos = [p * random.random() for p in self.pos]

# Boo hoo https://medium.com/@yasufumy/python-multiprocessing-c6d54107dd55
_func = None
def worker_init(func):
  global _func
  _func = func
def worker(x):
  return _func(x)
def xmap(func, iterable, processes=None):
  with Pool(processes, initializer=worker_init, initargs=(func,)) as p:
    return p.map(worker, iterable)

class PSO:
  def __init__(self,
               populationSize,
               nDimensions,
               createNewIndividual,
               fitness,
               wSchedule = lambda x: 0.9,
               c1 = 2,
               c2 = 2
  ):
    #self._populationSize = populationSize
    self._nDimensions = nDimensions
    self._createNewIndividual = createNewIndividual
    self._fitness = lambda x: self.wrapFitness(fitness, x)
    #self._fitness = fitness

    self._w = wSchedule
    self._c1 = c1
    self._c2 = c2

    self._iteration = 0

    self.particles = []

    self.best = (None, 0)

    for i in range(populationSize):
      position = Vector(createNewIndividual())
      velocity = Vector([0 for i in range(nDimensions)])
      score = self._fitness(position.pos)

      self.particles.append((position, velocity, score, position, score, i))

      if score > self.best[1]:
        self.best = (position, score)

  def wrapFitness(self, fxn, individual):
    try:
      return fxn(individual)
    except Exception as e:
      # Most likely one waycircle got enclosed in another
      # XXX compensate for this somehow?
      #print('Individual failed', individual)
      #print(e)
      return 0

  def innerLoop(self, particle):
    w = self._w(self._iteration)
    globalPosition, globalScore = self.best
    position, velocity, score, bestPosition, bestScore, id = particle
    toBest = bestPosition - position
    toGlobal = globalPosition - position

    
    toBest.perturb()
    toGlobal.perturb()

    gamma = 0.01

    vInertia = velocity * w
    vCognitive = gamma * self._c1 * toBest
    vSocial = gamma * self._c2 * toGlobal

    # Get the new stats
    newVelocity = vInertia + vCognitive + vSocial
    newPosition = position + newVelocity
    newScore = self._fitness(newPosition.pos)

    # Update local/global fitness
    if newScore > bestScore:
      bestScore = newScore
      bestPosition = newPosition

    return (newPosition, newVelocity, newScore, bestPosition, bestScore, id)
  

  def iterate(self, processes = 1, loopFun = lambda x, **kwargs: x):
    
    self._iteration += 1

    if processes > 1:
      results = xmap(lambda p: self.innerLoop(p), self.particles, processes=processes)
    else:
      results = [self.innerLoop(p) for p in self.particles]

    newParticles = sorted(results, key=lambda p: p[2])

    best = newParticles[-1]
    self.best = (best[0], best[2])
    self.particles = newParticles


    
