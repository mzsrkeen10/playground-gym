# -*- coding: utf-8 -*-

import numpy as np
import gym
import time


class MountainCarContinuous:

    def __init__(self, crossover_rate, mutation_rate, num=100):
        self.env = gym.make("MountainCarContinuous-v0")
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.num = num
        self.population = np.random.rand(num, 2) * 20 - 10
        self.fitness = np.zeros(num)
        for i, individual in enumerate(self.population):
            self.fitness[i] = self.calc_fitness(individual)

    def decide_action(self, individual, observation):
        obs = np.array(observation)
        s = np.tanh(individual.dot(obs))
        return s

    def calc_fitness(self, individual, flag=False):
        observation = self.env.reset()
        fitness = 0
        for t in range(1000):
            if flag:
                self.env.render()
                time.sleep(0.005)
            action = self.decide_action(individual, observation)
            observation, reward, done, info = self.env.step([action])
            fitness += reward
            if done:
                break
        return fitness if fitness > 0 else 0

    def one_point_crossover(self):
        np.random.shuffle(self.population)
        for i in range(0, self.num-1, 2):
            if np.random.rand() < self.crossover_rate:
                work = self.population[i][0]
                self.population[i][0] = self.population[i+1][0]
                self.population[i+1][0] = work

    def mutation(self):
        for individual in self.population:
            for i in range(2):
                if np.random.rand() < self.mutation_rate:
                    individual[i] = np.random.rand() * 20 - 10

    def roulette_selection(self):
        sum_fitness = sum(self.fitness)
        roulette = np.zeros(self.num)
        ac_roulette = np.zeros(self.num)
        if sum_fitness != 0:
            roulette[0] = self.fitness[0] / sum_fitness
            ac_roulette[0] = roulette[0]
            for i in range(1, self.num):
                roulette[i] = self.fitness[i] / sum_fitness
                ac_roulette[i] = ac_roulette[i-1] + roulette[i]
        new_population = np.random.rand(self.num, 2) * 20 - 10
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
    mcc = MountainCarContinuous(0.5, 0.05, 20)
    print("initial population")
    mcc.print_population()
    mcc.calc_fitness(mcc.population[np.argmax(mcc.fitness)], True)

    max_generation = 10
    for i in range(1, max_generation+1):
        mcc.one_point_crossover()
        mcc.mutation()
        for j in range(mcc.num):
            mcc.fitness[j] = mcc.calc_fitness(mcc.population[j])
        print("generation={}".format(i))
        mcc.print_population()
        mcc.calc_fitness(mcc.population[np.argmax(mcc.fitness)], True)
        mcc.roulette_selection()


if __name__ == "__main__":
    main()
