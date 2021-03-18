import random

from .BaseOptimizer import BaseOptimizer, Vector, xmap



class PSO(BaseOptimizer):
  def __init__(self,
               populationSize,
               nDimensions,
               createNewIndividual,
               fitness,
               wSchedule = lambda x: 0.9,
               c1 = 2,
               c2 = 2,
               bounds = None,
               **kwargs
  ):
    super().__init__(fitness, **kwargs)
    from math import inf

    #self._populationSize = populationSize
    self._nDimensions = nDimensions
    self._createNewIndividual = createNewIndividual
    #self._fitness = lambda x: self.wrapFitness(fitness, x)
    #self._fitness = fitness

    self._w = wSchedule
    self._c1 = c1
    self._c2 = c2
    self._bounds = bounds

    self._iteration = 0

    self.particles = []

    self.best = (None, -inf)

    if not hasattr(populationSize, '__iter__'):
      populationSize = range(populationSize)

    def createAndScore(data):
      i, pos = data
      position = Vector(pos)
      velocity = Vector([0 for i in range(nDimensions)])
      score = self._fitness(position.pos)

      #self.particles.append((position, velocity, score, position, score, i))
      return (position, velocity, score, position, score, i)

    print('starting')
    self.particles = xmap(createAndScore, [(i, createNewIndividual(i)) for i in populationSize], processes=self._processes)
    print('Generated initial %d particles' % len(self.particles))

    for particle in self.particles:
      position, vel, score, bpos, bscore, i = particle

      if score > self.best[1]:
        self.best = (position, score)

    if self.best[0] is None:
      print([p[2] for p in self.particles])
      raise TypeError('No initial gradient, did everyone fail?')
    
  def getIndividuals(self):
    return [p[0] for p in self.particles]

  def prepare(self):
    # Find our best particle
    globalPosition, globalScore = self.best

    # Update everyone
    w = self._w(self._iteration)
    newParticles = []
    for position, velocity, score, bestPosition, bestScore, idx in self.particles:
      toBest = bestPosition - position
      toGlobal = globalPosition - position

      
      toBest.perturb()
      toGlobal.perturb()

      gamma = 0.1

      vInertia = velocity * w
      vCognitive = self._c1 * toBest
      vSocial = self._c2 * toGlobal

      # Get the new stats
      newVelocity = vInertia + vCognitive + vSocial
      newPosition = position + gamma * newVelocity

      # Constrain, if we have bounds
      if self._bounds is not None:
        newPosition, newVelocity = self.bound(newPosition, newVelocity)

      # Score will be filled in in a moment
      newParticles.append((newPosition, newVelocity, score, bestPosition, bestScore, idx))

    self.particles = newParticles

    #print('bestie is', globalScore)
    return self.getBest()

  def bound(self, pos, vel):
    """
    Modify pos to be within our (possibly repeating) bounds.
    If a violation is detected, the velocity is also set to zero
    """
    for i in range(self._nDimensions):
      bound = self._bounds[i % len(self._bounds)]
      if len(bound) != 2:
        raise TypeError('Bounds should have an upper and lower')

      lower, upper = bound

      if lower is not None and pos[i] < lower:
        pos[i] = lower
        if vel[i] < 0:
          vel[i] = 0

      if upper is not None and pos[i] > upper:
        pos[i] = upper
        if vel[i] > 0:
          vel[i] = 0
    
    return pos, vel

  def update(self, fitnesses):
    if len(self.particles) != len(fitnesses):
      raise TypeError('Incorrect fitness vector')

    newParticles = []
    for i in range(len(self.particles)):
      position, velocity, score, bestPosition, bestScore, idx = self.particles[i]
      newScore = fitnesses[i]

      if newScore > bestScore:
        #print('  ', position, newScore, '>', bestPosition, bestScore)
        bestScore = newScore
        bestPosition = position
      
      newParticles.append((position, velocity, newScore, bestPosition, bestScore, idx))

    # Store the updated particles
    self.particles = sorted(newParticles, key=lambda p: p[2])

    # Check best
    newBest = self.getBest()
    if newBest[1] > self.best[1]:
      #print('new bestie', self.best, newBest, self._fitness(newBest[0]))
      self.best = newBest

  def getBest(self):
    bestPos, highScore = None, 0

    for position, velocity, score, bestPosition, bestScore, idx in self.particles:
      if bestPos is None or score > highScore:
        bestPos = position
        highScore = score

    return bestPos, highScore
  
  def snapshot(self, o, n):
    o('PSOv2:' + str(n) + ':' + str(self.best[1]))
    o([particle[2] for particle in self.particles])
    particles = self.particles[-n:]
    for particle in particles:
      o(particle[2])
      o(particle[0].pos)

  def dump(self, o):
    o('PSOv2final:' + str(len(self.particles)) + ':' + str(self.best[1]))
    for particle in self.particles:
      # position, velocity, score, bestPosition, bestScore, idx
      # ->
      # score, position, velocity, bestPosition, bestScore
      o(particle[2])
      o(particle[0].pos)
      o(particle[1].pos)
      o(particle[3].pos)
      o(particle[4])

  