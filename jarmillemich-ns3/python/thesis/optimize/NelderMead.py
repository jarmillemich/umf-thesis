from .BaseOptimizer import BaseOptimizer, Vector, xmap
import heapq

class VertexWrapper:
  def __init__(self, vec, fitness):
    self.vec = vec
    self.fitness = fitness
  
  def __lt__(self, other):
    return self.fitness < other.fitness

def centroid(vertices):
  return sum(vertices) / len(vertices)

# NB we are not really inheriting from BaseOptimizer because we are serial :(
class NelderMead(BaseOptimizer):
  def __init__(self, nDimensions, createVertex, fitness, preverts = None):
    super().__init__(fitness)
    
    if preverts is None:
      vertices = [Vector(createVertex()) for i in range(nDimensions + 1)]

      print('Judging vertices')
      self.vertices = xmap(lambda vec: VertexWrapper(vec, fitness(vec)), vertices, processes=24)
      print('Got initial judgement')
    else:
      if len(preverts) != nDimensions + 1:
        raise TypeError('Pre-populated vertices must be the correct length')

      print('Got prepopulated vertex array, cool!')
      self.vertices = preverts

    heapq.heapify(self.vertices)
    
    measure = sorted(self.vertices)
    print('Initial range is %.2f-%.2f' % (measure[0].fitness, measure[-1].fitness))

    self.bestScore = measure[-1].fitness

    self.coeff = (1, 2, 0.5, 0.5)

  # Override because serial
  def iterate(self):
    alpha, gamma, rho, sigma = self.coeff
    actualVertices = [el.vec for el in self.vertices[1:]]
    centr = centroid(actualVertices[-100:])
    
    worstPair = heapq.nsmallest(2, self.vertices)
    worstVec = worstPair[0].vec
    worstScore = worstPair[0].fitness

    secondWorstVec = worstPair[1].vec
    secondWorstScore = worstPair[1].fitness
    
    # Reflection
    reflectPoint = centr + alpha * (centr - worstVec)
    reflectFitness = self._fitness(reflectPoint)

    if reflectFitness > secondWorstScore and reflectFitness < self.bestScore:
      heapq.heapreplace(self.vertices, VertexWrapper(reflectPoint, reflectFitness))
      print('reflect simple %.2f->%.2f' % (worstScore, reflectFitness))
      return

    # Expansion (reflected was the best)
    if reflectFitness > self.bestScore:
      expandPoint = centr + gamma * (reflectPoint - centr)
      expandFitness = self._fitness(expandPoint)

      # Keep going in that direction, if it's better
      if expandFitness > reflectFitness:
        heapq.heapreplace(self.vertices, VertexWrapper(expandPoint, expandFitness))
        print('exand %.2f->%.2f' % (worstScore, expandFitness))
      # Stop at the reflected point
      else:
        heapq.heapreplace(self.vertices, VertexWrapper(reflectPoint, reflectFitness))
        print('reflect woops %.2f->%.2f' % (worstScore, reflectFitness))
      return
    
    # Contraction (reflected is still the worst, move towards the other vertices)
    contractPoint = centr + rho * (secondWorstVec - centr)
    contractFitness = self._fitness(contractPoint)

    if contractFitness > worstScore:
      heapq.heapreplace(self.vertices, VertexWrapper(contractPoint, contractFitness))
      print('contract %.2f->%.2f' % (worstScore, contractFitness))
      return

    # Shrink
    # Nothing got better, move everyone towards the best point
    raise TypeError('TODO')


