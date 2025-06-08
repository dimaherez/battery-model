import numpy as np
import random
import matplotlib.pyplot as plt

class GeneticAlgorithm:
    def __init__(self, pop_size=100, generations=100, mutation_rate=0.8):
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.fitness_history = []  # Stores best fitness per generation
        self.lower_bounds = np.full(31, 1e-3)
        self.upper_bounds = np.full(31, 1.0)
        self.initial_guess = np.random.uniform(0.001, 1.0, 31)


    def evaluate(self, population, df, C, fitness_function, isCharging):
        return [fitness_function(ind, df, C, isCharging) for ind in population]

    def select_parents(self, population, fitness, tournament_size=3):
        parents = []
        for _ in range(len(population) // 2):
            candidates = random.sample(list(zip(population, fitness)), tournament_size)
            winner = min(candidates, key=lambda x: x[1])[0]
            parents.append(winner)
        return parents


    def crossover(self, parents):
        children = []
        while len(children) + len(parents) < self.pop_size:
            p1, p2 = random.sample(parents, 2)
            pt = len(p1) // 2
            child = np.maximum(np.concatenate([p1[:pt], p2[pt:]]), 0)
            children.append(child)
        return parents + children

    def mutate(self, population, gen=None):
        scale = 0.1 if gen is None else max(0.01, 0.1 * (1 - gen / self.generations))
        for ind in population:
            if random.random() < self.mutation_rate:
                num_genes = 5
                for _ in range(num_genes):
                    i = random.randint(0, len(ind) - 1)
                    ind[i] = max(0, ind[i] + np.random.normal(0, scale))
        return population


    def optimize(self, df, C, fitness_function, isCharging, plot=True):
        population = self.initial_guess
        count = 1
        for gen in range(self.generations):
            fitness = self.evaluate(population, df, C, fitness_function, isCharging)
            best_fitness = min(fitness)  # lower is better (assumes a loss function)
            self.fitness_history.append(best_fitness)

            print(f"Generation: {count}; Best fitness: {best_fitness:.4f}")
            count += 1

            parents = self.select_parents(population, fitness)
            population = self.crossover(parents)
            population = self.mutate(population, gen)

        if plot:
            plt.plot(self.fitness_history, label='Best Fitness')
            plt.xlabel('Generation')
            plt.ylabel('Fitness (lower is better)')
            plt.title('Genetic Algorithm Optimization Progress')
            plt.grid(True)
            plt.legend()
            plt.show()

        return min(population, key=lambda x: fitness_function(x, df, C, isCharging))
    
    def init_generation(self, base_individual):
        self.initial_guess = []
        for _ in range(100):
            mutated_individual = base_individual + np.random.uniform(-0.5, 0.5, len(base_individual))
            mutated_individual = np.maximum(mutated_individual, 0)  # Prevent negative values
            self.initial_guess.append(mutated_individual)
