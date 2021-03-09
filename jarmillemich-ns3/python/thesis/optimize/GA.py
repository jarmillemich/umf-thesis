from .BaseOptimizer import BaseOptimizer, Vector, xmap
from .Genetics import Chromosome
import random

class GA(BaseOptimizer):
  def __init__(self,
               populationSize,
               nDimensions,
               mapping,
               fitness,
               # Probability of crossover occurring
               pCrossover = 0.5,
               # Percentage of bits which will flip in any child
               pMutation = 0.001,
               # Percentage of individuals which survive, (take top n%)
               pSurvival = 0.5,
               elitism = 2,
               **kwargs
  ):
    super().__init__(fitness, **kwargs)

    self.nDimensions = nDimensions
    self._mapping = mapping

    # How many individuals to have
    self._populationSize = populationSize
    
    self._pCrossover = pCrossover
    self._pMutation = pMutation
    self._pSurvival = pSurvival
    self._elitism = elitism

    bits = 16

    self.population = [
      Chromosome(nDimensions * bits)
      for i in range(populationSize)
    ]

  def prepare(self):
    pass

  def chromoToVector(self, chromo):
    ret = []

    for i in range(self.nDimensions):
      lo, hi = self._mapping[i]

      ret.append(chromo.getReal16(i * 16, lo, hi))

    return ret
  
  def getIndividuals(self):
    return [self.chromoToVector(ind) for ind in self.population]

  def weightedSelect(self, individuals, sumFitness):
    idx = random.random() * sumFitness
    total = 0
    for individual, fitness in individuals:
        total += fitness
        if idx < total:
            return individual

    raise IndexError('Random weighted select failed with %.2f/%.2f' % (idx, sumFitness))

  def update(self, fitnesses):
    self.lastFitness = sorted(fitnesses)
    individuals = [
      (self.population[i], fitnesses[i])
      for i in range(len(self.population))
    ]

    # Get sorted by fitness, highest first
    individuals.sort(key = lambda x: -x[1])

    # Next generation, pulling out elites first
    newIndividuals = [ind[0] for ind in individuals[:self._elitism]]
    survivors = individuals[:int(len(individuals)*self._pSurvival)]
    sumFitness = sum([f for i, f in survivors])

    # Create other individuals
    for i in range(self._elitism, self._populationSize, 2):
        leftParent = self.weightedSelect(individuals, sumFitness)
        rightParent = self.weightedSelect(individuals, sumFitness)

        leftChild = leftParent.clone()
        rightChild = rightParent.clone()

        # Perform crossover
        if random.random() < self._pCrossover:
            leftChild.randomSinglePointCrossover(rightChild)
        
        # Perform mutation
        leftChild.mutate(self._pMutation)
        rightChild.mutate(self._pMutation)

        # Add to next generation
        newIndividuals.append(leftChild)
        newIndividuals.append(rightChild)

    self.population = newIndividuals