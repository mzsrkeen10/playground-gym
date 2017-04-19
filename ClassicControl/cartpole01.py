# -*- coding: utf-8 -*-

import numpy as np
import gym
import time


class CartPole:

    def __init__(self, crossover_rate, mutation_rate, num=100):
        self.env = gym.make("CartPole-v1")
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.num = num
        self.population = np.random.rand(num, 4) * 2 - 1
        self.fitness = np.zeros(num)
        for i, individual in enumerate(self.population):
            self.fitness[i] = self.calc_fitness(individual)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def decide_action(self, individual, observation):
        obs = np.array(observation)
        s = self.sigmoid(individual.dot(obs))
        return 1 if s >= 0.5 else 0

    def calc_fitness(self, individual, flag=False):
        observation = self.env.reset()
        fitness = 0
        for t in range(10000):
            if flag:
                self.env.render()
                time.sleep(0.005)
            action = self.decide_action(individual, observation)
            observation, reward, done, info = self.env.step(action)
            fitness += reward
            if done:
                break
        return fitness

    def one_point_crossover(self):
        np.random.shuffle(self.population)
        for i in range(0, self.num-1, 2):
            if np.random.rand() < self.crossover_rate:
                cross_point = np.random.randint(3) + 1
                for j in range(cross_point):
                    work = self.population[i][j]
                    self.population[i][j] = self.population[i+1][j]
                    self.population[i+1][j] = work

    def mutation(self):
        for individual in self.population:
            for i in range(4):
                if np.random.rand() < self.mutation_rate:
                    individual[i] = np.random.rand() * 2 - 1

    def roulette_selection(self):
        sum_fitness = sum(self.fitness)
        roulette = np.zeros(self.num)
        ac_roulette = np.zeros(self.num)
        roulette[0] = self.fitness[0] / sum_fitness
        ac_roulette[0] = roulette[0]
        for i in range(1, self.num):
            roulette[i] = self.fitness[i] / sum_fitness
            ac_roulette[i] = ac_roulette[i-1] + roulette[i]
        new_population = np.zeros([self.num, 4])
        for i in range(self.num):
            r = np.random.rand()
            for j in range(self.num):
                if r <= ac_roulette[j]:
                    new_population[i] = np.array(self.population[j])
                    break
        self.population = np.array(new_population)

    def print_population(self):
        for i, individual in enumerate(self.population):
            print("{0}, fitness={1}".format(individual, self.fitness[i]))



def main():
    cp = CartPole(0.5, 0.05, 10)
    print("initial population")
    cp.print_population()
    cp.calc_fitness(cp.population[np.argmax(cp.fitness)], True)

    max_generation = 20
    for i in range(1, max_generation+1):
        cp.one_point_crossover()
        cp.mutation()
        for j in range(cp.num):
            cp.fitness[j] = cp.calc_fitness(cp.population[j])
        print("generation={}".format(i))
        cp.print_population()
        cp.calc_fitness(cp.population[np.argmax(cp.fitness)], True)
        cp.roulette_selection()


if __name__ == "__main__":
    main()
