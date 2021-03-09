from math import sqrt
from multiprocessing import Pool
from queue import Queue
import random

from BaseOptimizer import BaseOptimizer, Vector



class PSO(BaseOptimizer):
  def __init__(self,
               populationSize,
               nDimensions,
               createNewIndividual,
               fitness,
               wSchedule = lambda x: 0.9,
               c1 = 2,
               c2 = 2
  ):
    super().__init__(self)

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

    if not hasattr(populationSize, '__iter__'):
      populationSize = range(populationSize)

    def createAndScore(i):
      position = Vector(createNewIndividual())
      velocity = Vector([0 for i in range(nDimensions)])
      score = self._fitness(position.pos)

      #self.particles.append((position, velocity, score, position, score, i))
      return (position, velocity, score, position, score, i)

    print('starting')
    self.particles = xmap(createAndScore, populationSize, processes=16)
    print('Generated initial %d particles' % len(self.particles))

    for particle in self.particles:
      position, vel, score, bpos, bscore, i = particle

      if score > self.best[1]:
        self.best = (position, score)

    # for i in populationSize:
    #   position = Vector(createNewIndividual())
    #   velocity = Vector([0 for i in range(nDimensions)])
    #   score = self._fitness(position.pos)

    #   self.particles.append((position, velocity, score, position, score, i))

    #   if score > self.best[1]:
    #     self.best = (position, score)



    if self.best[0] is None:
      print([p[2] for p in self.particles])
      raise TypeError('No initial gradient, did everyone fail?')
    

  def wrapFitness(self, fxn, individual):
    try:
      return fxn(individual)
    except Exception as e:
      # Most likely one waycircle got enclosed in another
      # XXX compensate for this somehow?
      #print('Individual failed', individual)
      print(e)
      #print(e.stack)
      return 0

  def innerLoop(self, args):
    #print(self._iteration, self.best)
    particle, best = args
    w = self._w(self._iteration)
    globalPosition, globalScore = best
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

    ## EXPERIMENT when our velocity gets small, run away!
    if newVelocity.length() > 0 and newVelocity.length() < 20 and False:
      print('bouncer!', newVelocity.length())
      newPosition = Vector(self._createNewIndividual())

    return (newPosition, newVelocity, newScore, bestPosition, bestScore, id)
  

  def iterate(self, processes = 1, pool = None, **kwargs):
    
    self._iteration += 1

    if processes > 1 or pool is not None:
      # Beware, if we use a shared pool self is copied, not referenced!
      # TODO reduce the spookiness of this mp call
      results = xmap(lambda p: self.innerLoop(p, self.best), [(p, self.best) for p in self.particles], processes=processes, pool=pool)
    else:
      results = [self.innerLoop((p, self.best)) for p in self.particles]

    newParticles = sorted(results, key=lambda p: p[2])

    best = newParticles[-1]
    self.best = (best[0], best[2])
    self.particles = newParticles


  def iterateMany(self, iterations = 1, processes = 1, cb = lambda x: None):
    if type(iterations) == int:
      iterations = range(iterations)

    # Use a single pool, this saves several hundred ms per loop
    with Pool(processes, initializer=worker_init, initargs=(lambda x: self.innerLoop(x),)) as p:
      for i in iterations:
        #worker_init(lambda x: self.innerLoop(x))
        self.iterate(pool = p)
        #self.iterate(processes = 24)
        

        cb(i)