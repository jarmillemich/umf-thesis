from math import sqrt
from multiprocessing import Pool
import random
import tqdm


class Vector:
  def __init__(self, pos):
    self.pos = [float(e) for e in pos]

  def __len__(self):
    return len(self.pos)

  def __add__(self, other):
    if len(self.pos) != len(other.pos):
      raise TypeError('Vectors must be of the same length to __add__ %d vs %d' % (len(self.pos), len(other.pos)))

    newPos = self.pos.copy()
    for i in range(len(self.pos)):
      newPos[i] += other.pos[i]

    return Vector(newPos)

  def __radd__(self, other):
    if other == 0:
      # To work with sum(), e.g.
      return self
    return other + self

  def __sub__(self, other):
    if len(self.pos) != len(other.pos):
      raise TypeError('Vectors must be of the same length to __sub__')

    newPos = self.pos.copy()
    for i in range(len(self.pos)):
      newPos[i] -= other.pos[i]
    
    return Vector(newPos)

  def __mul__(self, other):
    if isinstance(other, Vector):
      return self.dot(other)
    else:
      # Hopefully multiplication by a scalar
      scalar = other
    
      newPos = self.pos.copy()
      for i in range(len(self.pos)):
        newPos[i] *= scalar
      
      return Vector(newPos)

  def __rmul__(self, scalar):
    return self * scalar

  def dot(self, other):
    if len(self) != len(other):
      raise TypeError('Mismatched dimensions')

    total = 0
    for i in range(len(self)):
      total += self[i] * other[i]

    return total

  def __truediv__(self, scalar):
    return self * (1 / scalar)

  def __repr__(self):
    if len(self.pos) < 10:
      return 'Vector<' + ','.join([str(round(p,2)) for p in self.pos]) + '>'
    else:
      return 'Vector<' + ','.join([str(round(p,2)) for p in self.pos[:5]]) + ' ... ' + ','.join([str(round(p,2)) for p in self.pos[-5:]]) + '>'

  def length(self):
    sum = 0
    for p in self.pos:
      sum += p ** 2
    return sqrt(sum)

  def rotate(self, theta):
    if len(self) != 2:
      raise NotImplementedError('Rotations only supported on R2 vectors')
    
    from math import sin, cos
    c = cos(theta)
    s = sin(theta)

    return Vector([c * self[0] - s * self[1], s * self[0] + c * self[1]])

  def angle(self):
    if len(self) != 2:
      raise TypeError('Angles only supported on R2 vectors')
    from math import atan2
    return atan2(self[1], self[0])

  def perturb(self):
    r = random.random()
    # Comment/uncommet for element vs whole vector multiplication
    #self.pos = [p * r for p in self.pos]self.pos
    self.pos = [p * random.random() for p in self.pos]

  def __getitem__(self, key):
    # Pass through index access to array
    return self.pos[key]

  def __setitem__(self, key, value):
    # Pass through index access to array
    self.pos[key] = value

# Boo hoo https://medium.com/@yasufumy/python-multiprocessing-c6d54107dd55
_func = None
def worker_init(func):
  global _func
  _func = func
def worker(x):
  return _func(x)
def xmap(func, iterable, processes=None, pool = None):
  if pool is not None:
    return pool.map(worker, iterable)
  else:
    with Pool(processes, initializer=worker_init, initargs=(func,)) as p:
      return p.map(worker, iterable)
def tmap(func, iterable, processes=None, pool=None, leave=True):
  if pool is not None:
    return list(tqdm.auto.tqdm(pool.imap(worker, iterable), total=len(iterable), leave=leave))
  else:
    with Pool(processes, initializer=worker_init, initargs=(func,)) as p:
      return list(tqdm.auto.tqdm(p.imap(worker, iterable), total=len(iterable), leave=leave))

class BaseOptimizer:
  def __init__(self, fitness, processes=1):
    self._fitness = lambda x: self.wrapFitness(fitness, x)
    self._processes = processes
    
  def iterate(self, pool = None):
    state = self.prepare()
    # ONLY the fitness evaluation is multi-threaded, to save us from weird state things
    fitnesses = self.evaluateGeneration(state, pool=pool)
    self.update(fitnesses)

  def iterateMany(self, iterations = 1, processes = 1, cb = lambda x: None):
    # We could pass in a number, or perhaps a tqdm
    if type(iterations) == int:
      iterations = range(iterations)

    # Use a single pool, this saves several hundred ms per loop
    if processes > 1:
      # Using this hack as we seem to be leaking memory somewhere, TODO track that down
      # It seems like .map will do ~4 batches per iteration
      batchSize = 128
      runs = 0
      for batch in range(iterations / batchSize):
        with Pool(processes, initializer=worker_init, initargs=(self._fitness,)) as p:
          for i in range(batchSize):
            self.iterate(pool = p)
            cb(i)

            runs += 1
            if runs >= iteratons:
              break
    else:
      for i in iterations:
        self.iterate()
        cb(i)

  def getIndividuals(self):
    raise NotImplementedError('BaseOptimizer children must define getIndividuals(): Vector[]')

  def wrapFitness(self, fxn, individual):
    try:
      return fxn(individual)
    except Exception as e:
      import traceback
      # Most likely one waycircle got enclosed in another
      # XXX compensate for this somehow?
      #print('Individual failed', individual)
      #print('Failure', e)
      #traceback.print_exc()
      #print(e.stack)
      # Hopefully this is the worst
      return -9e9

  def evaluateGeneration(self, state, pool = None):
    # Compute the fitness for each vector
    individuals = self.getIndividuals()

    if self._processes > 1 or pool is not None:
      results = xmap(lambda p: self._fitness(p), [(p) for p in individuals], processes=self._processes, pool=pool)
    else:
      results = [self._fitness(p) for p in individuals]

    return results