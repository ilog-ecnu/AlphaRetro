import numpy as np

class GeneticAlgorithm:
    def __init__(self, population_size, genome_length, beam_size, crossover_rate, mutation_rate):
        self.population_size = population_size
        self.genome_length = genome_length
        self.beam_size = beam_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
    
    def init_population(self):
        """初始化种群，每个基因位点的值为0到10之间的整数"""
        # np.random.seed(4874)
        return np.random.randint(0, self.beam_size, size=(self.population_size, self.genome_length))
    
    def crossover(self, parent1, parent2):
        """交叉操作，单点交叉"""
        if np.random.rand() < self.crossover_rate:
            point = np.random.randint(1, self.genome_length)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            return child1, child2
        return parent1.copy(), parent2.copy()
    
    def mutate(self, individual):
        """变异操作，每个基因位点随机变异，增加或减少1，或不变，确保值在0到10之间"""
        for i in range(self.genome_length):
            if np.random.rand() < self.mutation_rate:
                change = np.random.choice([-1, 0, 1])
                individual[i] = np.clip(individual[i] + change, 0, self.beam_size-1)  # np.clip是闭区间，右半边不能取，因此 -1
        return individual
    
    def create_new_generation(self, population):
        """创建新一代种群"""
        new_population = []
        indices = np.random.permutation(self.population_size)
        for i in range(0, self.population_size, 2):
            parent1 = population[indices[i]]
            parent2 = population[indices[i+1]]
            child1, child2 = self.crossover(parent1, parent2)
            new_population.append(self.mutate(child1))
            new_population.append(self.mutate(child2))
        return np.array(new_population[:self.population_size])
    
    def run(self):
        """运行一代遗传算法"""
        population = self.init_population()
        new_population = self.create_new_generation(population)
        return new_population
    
class GA_branch:
    def __init__(self, population_size, genome_length, beam_size, crossover_rate, mutation_rate):
        self.population_size = population_size
        self.genome_length = genome_length
        self.beam_size = beam_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
    
    def init_population(self):
        """初始化种群，每个基因位点的值为0到10之间的整数"""
        # return np.random.randint(0, self.beam_size, size=(self.population_size, self.genome_length))
        return np.random.randint(0, self.beam_size, size=(self.population_size, self.genome_length))
    
    def crossover(self, parent1, parent2):
        """交叉操作，单点交叉"""
        if np.random.rand() < self.crossover_rate:
            point = np.random.randint(1, self.genome_length)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            return child1, child2
        return parent1.copy(), parent2.copy()
    
    def mutate(self, individual):
        """变异操作，每个基因位点随机变异，增加或减少1，或不变，确保值在0到10之间"""
        for i in range(self.genome_length):
            if np.random.rand() < self.mutation_rate:
                change = np.random.choice([-1, 0, 1])
                individual[i] = np.clip(individual[i] + change, 0, self.beam_size-1)
        return individual
    
    def mutate_flip(self, individual):
        """变异操作，每个基因位点根据突变率随机翻转"""
        for i in range(self.genome_length):
            if np.random.rand() < self.mutation_rate:
                # 如果当前位是0，则变为1；如果是1，则变为0
                individual[i] = 1 - individual[i]
        return individual
    
    def create_new_generation_flip(self, population):
        """创建新一代种群"""
        new_population = []
        indices = np.random.permutation(self.population_size)
        for i in range(0, self.population_size, 2):
            parent1 = population[indices[i]]
            parent2 = population[indices[i+1]]
            new_population.append(self.mutate_flip(parent1))
            new_population.append(self.mutate_flip(parent2))
        return np.array(new_population[:self.population_size])

    def create_new_generation(self, population):
        """创建新一代种群"""
        new_population = []
        indices = np.random.permutation(self.population_size)
        for i in range(0, self.population_size, 2):
            parent1 = population[indices[i]]
            parent2 = population[indices[i+1]]
            child1, child2 = self.crossover(parent1, parent2)
            new_population.append(self.mutate(child1))
            new_population.append(self.mutate(child2))
        return np.array(new_population[:self.population_size])
    
    def run(self):
        """运行一代遗传算法"""
        population = self.init_population()
        new_population = self.create_new_generation(population)
        return new_population