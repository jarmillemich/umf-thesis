import random, math

def toGray(v):
    return v ^ (v >> 1)

def fromGray(g):
    v = 0
    while g > 0:
        v ^= g
        g >>= 1
        
    return v

class Chromosome:
    def __init__(self, length):
        # Start out random
        self._length = length
        #random.seed(0)
        self.randomize()
        
    # Completely randomize our gene
    def randomize(self):
        self._gene = random.getrandbits(self._length)

    # Return a new Chromosome with the same gene
    def clone(self):
        ret = Chromosome(self._length)
        ret._gene = self._gene
        return ret
        
    # Swap the masked bits with the other chromosome
    def crossover(self, other, mask):
        if self._length != other._length:
            raise TypeError('Cannot splice non-equal length chromosomes')
            
        # Get some masks
        fullMask = 2**self._length - 1
        unMask = fullMask ^ mask
        
        # Get values to swap
        us = self._gene & mask
        them = other._gene & mask
        
        # Unset those bits
        self._gene &= unMask
        other._gene &= unMask
        
        # Set the others bits
        self._gene |= them
        other._gene |= us
        
    # Swap one side of our genome with another
    def singlePointCrossover(self, other, crossAt):
        mask = 2**crossAt - 1
        self.crossover(other, mask)
        
    # Crossover at a random point
    def randomSinglePointCrossover(self, other):
        pt = math.floor(self._length * random.random())
        self.singlePointCrossover(other, pt)
        
    # Mutate each bit with probability pM
    def mutate(self, pM):
        for i in range(self._length):
            if random.random() < pM:
                self._gene ^= 2**i
    
    # Get a binary string from our gene
    def getRawValue(self, idx, bits):
        mask = (1 << bits) - 1
        return (self._gene >> idx) & mask
    
    # Set a binary string in our gene
    def setRawValue(self, idx, bits, data):
        mask = (1 << bits) - 1
        # It should match...
        data &= mask
        # Can't really do an &~data if we don't know the total size
        toRemove = self._gene & (mask << idx)
        self._gene -= toRemove
        self._gene |= data << idx
        
    # Integer getters
    def _getUint(self, idx, bits, gray):
        raw = self.getRawValue(idx, bits)
        return fromGray(raw) if gray else raw
    
    def getUint8(self, idx, gray = True):
        return self._getUint(idx, 8, gray)
    
    def getUint16(self, idx, gray = True):
        return self._getUint(idx, 16, gray)
    
    def getUint32(self, idx, gray = True):
        return self._getUint(idx, 32, gray)
    
    # Real getters
    def _getReal(self, idx, bits, lower, upper, gray):
        raw = self._getUint(idx, bits, gray)
        nRange = upper - lower
        nDelta = 1.0 * raw / (1<<bits)
        return lower + nDelta * nRange
    
    def getReal8(self, idx, lower = 0.0, upper = 1.0, gray = True):
        return self._getReal(idx, 8, lower, upper, gray)
    
    def getReal16(self, idx, lower = 0.0, upper = 1.0, gray = True):
        return self._getReal(idx, 16, lower, upper, gray)
    
    def getReal32(self, idx, lower = 0.0, upper = 1.0, gray = True):
        return self._getReal(idx, 32, lower, upper, gray)
        
    
    
# Borrowed from our previous work found at http://homepages.umflint.edu/~jarmille/CSC546proj/vFinal/client/AI/GARunner.ts
class GARunner:
    def __init__(self,
                 populationSize, 
                 createNewIndividual,
                 fitness,
                 # Probability of crossover occurring
                 pCrossover = 0.5,
                 # Percentage of bits which will flip in any child
                 pMutation = 0.001,
                 # Percentage of individuals which survive, (take top n%)
                 pSurvival = 0.5,
                 elitism = 2
                ):
        # How many individuals to have
        self._populationSize = populationSize
        # Function to create a new individual
        self._createNewIndividual = createNewIndividual
        # Function to evaluate fitness of an individual
        self._fitness = lambda x: self.wrapFitness(fitness, x)

        self._pCrossover = pCrossover
        self._pMutation = pMutation
        self._pSurvival = pSurvival
        self._elitism = elitism

        self.population = [
            self._createNewIndividual()
            for i in range(self._populationSize)
        ]

    def wrapFitness(self, fxn, individual):
        try:
            return fxn(individual)
        except:
            # Most likely one waycircle got enclosed in another
            # XXX compensate for this somehow?
            #print('Individual failed')
            return 0

    def weightedSelect(self, individuals, sumFitness):
        idx = random.random() * sumFitness
        total = 0
        for individual, fitness in individuals:
            total += fitness
            if idx < total:
                return individual

        raise IndexError('Random weighted select failed with %.2f/%.2f' % (idx, sumFitness))

    def iterate(self, loopFun = lambda x, **kwargs: x):
        # Evaluate everyone
        individuals = [
            (individual, self._fitness(individual))
            for individual
            in loopFun(self.population)
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
        return [f for i, f in individuals]

