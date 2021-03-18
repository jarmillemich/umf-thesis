from .BaseOptimizer import BaseOptimizer, Vector, xmap, tmap
import tqdm

class VertexWrapper:
  def __init__(self, vec, fitness):
    self.vec = vec
    self.fitness = fitness
  
  def __lt__(self, other):
    return self.fitness < other.fitness

def centroid(vertices):
  return sum(vertices) / len(vertices)

# REFLECT was the best
OP_REFLECT = 0
# EXPAND was the best
OP_EXPAND = 1
# CONTRACT was the best
OP_CONTRACT = 2
# No improvement. Will trigger a SHRINK iff all the vertices in this iteration do the same
OP_NONE = 3

class ParallelNelderMead(BaseOptimizer):
  def __init__(self, nDimensions, createVertex, fitness, k = 1, initVerts = None, adaptive = False, **kwargs):
    super().__init__(self.multiFitness(fitness), **kwargs)
    self.__rawFitness = fitness
    print('TEST', self.__rawFitness.evaluations)
    self.k = k

    self.nDimensions = nDimensions
    
    if initVerts is None:
      self._initverts = vertices = [Vector(createVertex(i)) for i in range(nDimensions + 1)]

      print('Judging vertices (%d)' % len(vertices))
      # Moderately awful
      if self.nDimensions < 1000:
        self.vertices = tmap(lambda vec: VertexWrapper(vec, fitness(vec)), vertices, processes=self._processes)
      else:
        self.vertices = []
        for start_idx in tqdm.auto.tqdm(range(0, self.nDimensions, 1000)):
          self.vertices.extend(tmap(lambda vec: VertexWrapper(vec, fitness(vec)), vertices[start_idx:start_idx+1000], processes=self._processes, leave=False))
      print('Got initial judgement')
    else:
      # Cheating mode (prescored vertices)
      if len(initVerts) != nDimensions + 1:
        raise TypeError('Pre-populated vertices must be the correct length')

      print('Got prepopulated vertex array, cool!')
      self.vertices = initVerts

    self.vertices = sorted(self.vertices)
    
    print('Initial range is %.2f-%.2f' % (self.vertices[0].fitness, self.vertices[-1].fitness))

    self.bestScore = self.vertices[-1].fitness

    if adaptive:
      # https://link.springer.com/content/pdf/10.1007/s10589-010-9329-3.pdf
      self.coeff = (1, 1 + 2 / nDimensions, 0.75 - 1 / (2 * nDimensions), 1 - 1 / nDimensions)
    else:
      self.coeff = (1, 2, 0.5, 0.5)

  def multiFitness(self, fitness):
    # Fitness in the parallel map, implements the core of NM
    def inner(item, state=None):
      try:
        reflect, expand, contract, cutoff, best = item

        reflectFitness = fitness(reflect)

        if reflectFitness > cutoff and reflectFitness < best:
          return (OP_REFLECT, reflectFitness)
        
        if reflectFitness > best:
          expandFitness = fitness(expand)

          if expandFitness > reflectFitness:
            return (OP_EXPAND, expandFitness)
          else:
            return (OP_REFLECT, reflectFitness)

        contractFitness = fitness(contract)

        if contractFitness > cutoff:
          return (OP_CONTRACT, contractFitness)
      except Exception as e:
        #print('Failure', e)
        pass

      # We can have negative fitness, so we don't want to reward a 0...
      # Just take the lowest we have and go down a notch, that'll never diverge!
      return (OP_NONE, min([v.fitness for v in self.vertices]))
  
    # XXX
    inner.evaluations = fitness.evaluations

    return inner

  def prepare(self):
    k = self.k

    # Split the vertices into those that will update and those that stay the same
    updateVertices = self.vertices[:k]
    simpleVertices = self.vertices[k:]
    cutoff = self.vertices[k].fitness
    best = self.vertices[-1].fitness

    # Compute the centroid of the non-update vertices
    center = centroid([v.vec for v in simpleVertices])

    self.iterationVertices = [self.prepareVertex(v.vec, center, cutoff, best) for v in updateVertices]

  def update(self, results):
    k = self.k
    newVertices = self.vertices[k:]

    if all([r[0] == OP_NONE for r in results]):
      # Everyone wants to shrink
      print('SHRINK')
      self.shrink()
      return

    if len(results) != k:
      raise TypeError('Got back wrong number of results %d != %d' % (len(results), k))

    updatedVerts = []
    counts = [0,0,0,0]

    for i in range(k):
      vertex = self.vertices[i]
      reflectPoint, expandPoint, contractPoint, cutoff, best = self.iterationVertices[i]
      op, fitness = results[i]

      if op == OP_REFLECT:
        updatedVerts.append(VertexWrapper(reflectPoint, fitness))
      elif op == OP_EXPAND:
        updatedVerts.append(VertexWrapper(expandPoint, fitness))
      elif op == OP_CONTRACT:
        updatedVerts.append(VertexWrapper(contractPoint, fitness))
      elif op == OP_NONE:
        # Nothing to do, and we didn't all want to shrink, stay as is and hope for better luck next time
        updatedVerts.append(vertex)
      else:
        raise TypeError('Invalid operation code %s' % op)

      counts[op] += 1
  
    # Put it all back together again
    newVertices += updatedVerts
    self.vertices = sorted(newVertices)

    if len(self.vertices) != self.nDimensions + 1:
      raise TypeError('Leaking vertices? %d != %d' % (len(self.vertices), self.nDimensions + 1))

    print('  Did updates over RECN:', counts, self.vertices[-1].fitness)

  def shrink(self):
    alpha, gamma, rho, sigma = self.coeff

    # Shrink all vertices towards the best
    # The books exclude the best vertex, but there is no difference in the math (sigma * (v - v) = 0)
    best = self.vertices[-1].vec

    vecs = [v.vec for v in self.vertices]
    print('from', vecs[0], 'towards', best)
    vecs = [best + sigma * (v - best) for v in vecs]
    print('to', vecs[0])

    shrinkFitness = lambda x: self.wrapFitness(self.__realFitness, x)

    self.vertices = xmap(lambda vec: VertexWrapper(vec, shrinkFitness(vec)), vecs, processes=24)
    self.vertices = sorted(self.vertices)
    self.bestScore = self.vertices[-1].fitness

    print('and', self.vertices[0].vec)


  def getIndividuals(self):
    return self.iterationVertices

  def prepareVertex(self, vert, center, cutoff, best):
    alpha, gamma, rho, sigma = self.coeff
    reflectPoint = center + alpha * (center - vert)
    expandPoint = center + gamma * (reflectPoint - center)
    contractPoint = center + rho * (vert - center)

    # Put these together to be evaluated in parallel
    return (reflectPoint, expandPoint, contractPoint, cutoff, best)


  def snapshot(self, o, n):
    o('PNM:' + str(n) + ':' + str(self.vertices[-1].fitness))
    o([vertex.fitness for vertex in self.vertices])
    vertices = self.vertices[-n:]
    for vertex in vertices:
      o(vertex.fitness)
      o(vertex.vec.pos)

  def dump(self, o):
    o('PNMfinal:' + str(len(self.vertices)) + str(self.vertices[-1].fitness))
    for vertex in self.vertices:
      o(vertex.fitness)
      o(vertex.vec.pos)
