from Agents.RandomAgent import RandomAgent as ra
from Agents.AdrianHerasAgent import AdrianHerasAgent as aha
from Agents.AlexPastorAgent import AlexPastorAgent as apa
from Agents.AlexPelochoJaimeAgent import AlexPelochoJaimeAgent as apja
from Agents.CarlesZaidaAgent import CarlesZaidaAgent as cza
from Agents.CrabisaAgent import CrabisaAgent as ca
from Agents.EdoAgent import EdoAgent as ea
from Agents.PabloAleixAlexAgent import PabloAleixAlexAgent as paaa
from Agents.SigmaAgent import SigmaAgent as sa
from Agents.TristanAgent import TristanAgent as ta
import random

import numpy as np
import time
import concurrent.futures

from Managers.GameDirector import GameDirector

class GA():
    def __init__(self, ind_size, agents, rounds = 20):
        self.ind_size = ind_size
        self.agents = agents
        self.rounds = rounds
        self.population = []
        self.fitness = []
        self.others = [[1/self.ind_size for i in range(self.ind_size)] for j in range(3)]


    def game(self, all_agents):
        # Ejemplo de ejecución
        try:
            game_director = GameDirector(agents=all_agents, max_rounds=200, store_trace=False)
            game_trace = game_director.game_start(print_outcome=False)
            return game_trace
        except Exception as e:
            print(f"Error: {e}")
            return 0

    def create_individual(self):
        a = [random.random() for i in range(self.ind_size)]
        suma = sum(x for x in a)
        a = [x/suma for x in a]
        return a
    
    def create_population(self, pop_size):
        population = []
        for i in range(pop_size):
            population.append(self.create_individual())
        return population


    def evaluate_game_winner(self, chosen_agent, game_trace):
        # Análisis de resultados
        last_round = max(game_trace["game"].keys(), key=lambda r: int(r.split("_")[-1]))
        last_turn = max(game_trace["game"][last_round].keys(), key=lambda t: int(t.split("_")[-1].lstrip("P")))
        victory_points = game_trace["game"][last_round][last_turn]["end_turn"]["victory_points"]
        
        winner = max(victory_points, key=lambda player: int(victory_points[player]))
        #print(sorted([(int(v),int(k.lstrip("J"))) for k,v in victory_points.items()], reverse=True))
        if chosen_agent == int(winner.lstrip("J")):  
            return 1, int(winner.lstrip("J"))
        return 0, int(winner.lstrip("J"))
    
    def evaluate_game_order(self, game_trace):
        # Análisis de resultados
        last_round = max(game_trace["game"].keys(), key=lambda r: int(r.split("_")[-1]))
        last_turn = max(game_trace["game"][last_round].keys(), key=lambda t: int(t.split("_")[-1].lstrip("P")))
        victory_points = game_trace["game"][last_round][last_turn]["end_turn"]["victory_points"]
        
        winner = max(victory_points, key=lambda player: int(victory_points[player]))
        sorted_list = sorted([(int(v),int(k.lstrip("J"))) for k,v in victory_points.items()], reverse=True)
        game_order = [j for i, j in sorted_list]
        #print(game_order)
        return game_order

    def evaluate_individual(self, individual, verbose = False):

        player = list(np.random.choice(self.agents, size = 1, p = individual))
        others = list(np.random.choice(self.agents, size=3))
        idx = random.randint(0,3)
        order = others[:idx]+player+others[idx:]
        if verbose:
            print(f"Game: {others} vs {player}, \nOrder: {order} \nPlayer: {idx}")

        trace = self.game(order) 
        a = self.evaluate_game_winner(idx, trace)
        return a
    
    def evaluate_group(self, individuals, verbose = False):

        players = [(np.random.choice(self.agents, size = 1, p = individual)[0], i) for i, individual in enumerate(individuals)]
        random.shuffle(players)
        players, idx = zip(*players)
        players, idx = list(players), list(idx)
        if verbose:
            print(f"Players: {players}, \nOrder: {idx}")

        trace = self.game(players) 
        a = self.evaluate_game_winner(0, trace)
        return a[1], idx #a[1] is the index in the game (0to3) of the winner, and idx is the order of the individuals in the game (if idx[0]==1 it means that individual 1 had the first turn (turn 0))


    def evaluate_group_order(self, individuals, verbose = False):

        players = [(np.random.choice(self.agents, size = 1, p = individual)[0], i) for i, individual in enumerate(individuals)]
        random.shuffle(players)
        players, idx = zip(*players)
        players, idx = list(players), list(idx)
        if verbose:
            print(f"Players: {players}, \nOrder: {idx}")

        trace = self.game(players) 
        a = self.evaluate_game_order(trace)
        #indiv_order = [individuals[idx[i]] for i in a]
        return a, idx 
        #a is the order inside the game (0to3, a[0]==1 means that player with first turn ended in 2nd position)
        #idx is the order of the individuals in the game (if idx[0]==1 it means that individual 1 had the first turn (turn 0))



    def evaluate_population_tournament(self):
        pop = random.sample(self.population, k=len(self.population))
        selected = []
        new_fitness = []
        for i in range(0, len(pop)-3, 4):
            indivs = [pop[i], pop[i+1], pop[i+2], pop[i+3]]
            indiv_fitness = {0:0, 1:0, 2:0, 3:0}
            for j in range(self.rounds):
                winner_order, game_order = self.evaluate_group_order(indivs)
                for idx, k in enumerate(winner_order):
                    indiv_fitness[game_order[k]] += idx
            indivs_ordered = sorted([(v,k) for k,v in indiv_fitness.items()])
            new_fit, new_ind = indivs_ordered[0]
            new_fitness.append(new_fit/self.rounds)
            selected.append(indivs[new_ind])
        return selected, new_fitness
    

    def evaluate_population_tournament_parallel(self, len_selected = 1):
        pop = random.sample(self.population, k=len(self.population))
        selected = []
        new_fitness = []
        for i in range(0, len(pop)-3, 4):
            indivs = [pop[i], pop[i+1], pop[i+2], pop[i+3]]
            indiv_fitness = {0:0, 1:0, 2:0, 3:0}
            with concurrent.futures.ProcessPoolExecutor() as pool:
                for res in  zip(pool.map(self.evaluate_group_order, [indivs for j in range(self.rounds)])):
                    winner_order, game_order = res[0]
                    for idx, k in enumerate(winner_order):
                        indiv_fitness[game_order[k]] += idx
            indivs_ordered = sorted([(v/self.rounds,k) for k,v in indiv_fitness.items()])
            new_fit, new_ind = [f for f, _ in indivs_ordered[0:len_selected]], [f for _, f in indivs_ordered[0:len_selected]]
            new_fitness = new_fitness + new_fit
            selected = selected + [indivs[n] for n in new_ind]
        return selected, new_fitness
    
    def evaluate_initial_population_tournament_parallel(self):
        pop = random.sample(self.population, k=len(self.population))
        fitness = []
        #individuals = []
        for i in range(0, len(pop)-3, 4):
            indivs = [pop[i], pop[i+1], pop[i+2], pop[i+3]]
            indiv_fitness = {0:0, 1:0, 2:0, 3:0}
            with concurrent.futures.ProcessPoolExecutor() as pool:
                for res in zip(pool.map(self.evaluate_group_order, [indivs for j in range(self.rounds)])):
                    winner_order, game_order = res[0]
                    for idx, k in enumerate(winner_order):
                        indiv_fitness[game_order[k]] += idx
            indivs_ordered = sorted([(k,v) for k,v in indiv_fitness.items()])
            fit = [v/self.rounds for k,v in indivs_ordered]
            fitness = fitness + fit
            #individuals = individuals + [indivs[k] for k,v in indivs_ordered]
        #print(all(x == y for x,y in zip(pop, individuals)))
        return pop, fitness
    
    def evaluate_population_tournament_from_existing_fitness(self, len_selected = 1):

        idx = random.sample(range(len(self.population)), k=len(self.population))
        pop = [self.population[i] for i in idx]
        fitness = [self.fitness[i] for i in idx]
        selected = []
        new_fitness = []
        for i in range(0, len(pop)-3, 4):
            indivs = [pop[i], pop[i+1], pop[i+2], pop[i+3]]
            indiv_fitness = {0:fitness[0], 1:fitness[1], 2:fitness[2], 3:fitness[3]}
            indivs_ordered = sorted([(v,k) for k,v in indiv_fitness.items()])
            new_fit, new_ind = [f for f, _ in indivs_ordered[0:len_selected]], [f for _, f in indivs_ordered[0:len_selected]]
            new_fitness = new_fitness + new_fit
            selected = selected + [indivs[n] for n in new_ind]
        return selected, new_fitness
    
    def evaluate_initial_population_individually_parallel(self):
        fitness = []
        min_fit = 10000
        best_indiv = -1
        for i in range(len(self.population)):
            indivs = [self.population[i]]+self.others
            indiv_fitness = {0:0, 1:0, 2:0, 3:0}
            with concurrent.futures.ProcessPoolExecutor() as pool:
                for res in zip(pool.map(self.evaluate_group_order, [indivs for j in range(self.rounds)])):
                    winner_order, game_order = res[0]
                    for idx, k in enumerate(winner_order):
                        indiv_fitness[game_order[k]] += idx/self.rounds
            fitness.append(indiv_fitness[0])
            if indiv_fitness[0] < min_fit:
                min_fit = indiv_fitness[0]
                best_indiv = self.population[i]
        return fitness, min_fit, best_indiv


    def crossover_avg(self, indiv1, indiv2):
        child =  [(x+y)/2 for x, y in zip(indiv1, indiv2)]
        suma = sum(child)
        return [x/suma for x in child]

    def crossover_onepoint(self, indiv1, indiv2):
        idx = random.randint(0, len(indiv1)-1)
        child1 = indiv1[:idx] + indiv2[idx:]
        child2 = indiv2[:idx] + indiv1[idx:]
        suma1 = sum(child1)
        suma2 = sum(child2)
        return [x/suma1 for x in child1], [x/suma2 for x in child2]

    def mutation_power(self, indiv, power = 2):
        suma = sum(x**power for x in indiv)
        return [(x**power)/suma for x in indiv]

    def mutation_twopoints(self, indiv):
        idx1 = random.randint(0, len(indiv)-2)
        idx2 = random.randint(idx1+1, len(indiv)-1)
        elem1 = indiv[idx1]
        elem2 = indiv[idx2]
        indiv[idx1] = elem2
        indiv[idx2] = elem1
        return indiv


    def selection(self, amount = None, tournament_size = 2):
        idx = random.sample(list(range(len(self.population))), k=len(self.population))
        selected = []
        if amount is None: amount = len(idx)
        for i in range(0, len(idx)-1, tournament_size):
            min_fit = 1000
            min_id = -1
            for j in range(tournament_size):
                if self.fitness[idx[i+j]] < min_fit:
                    min_fit = self.fitness[idx[i+j]]
                    min_id = idx[i+j]
            selected.append(min_id)
            if len(selected) >= amount:
                return selected
        return selected

    def __call__(self, pop_size = 20, generations = 20, tournament_size = 2):
        self.population = self.create_population(pop_size)
        for gen in range(generations):
            try:
                self.fitness, min_fit, self.best_indiv = self.evaluate_initial_population_individually_parallel()
            except: 
                self.fitness, min_fit, self.best_indiv = self.evaluate_initial_population_individually_parallel()
            self.min_fit = min_fit
            print(f"Generation {gen}. Best fitness: {min_fit:.3f}. Best individual: {[round(x, 3) for x in self.best_indiv]}, Mean fitness: {sum(self.fitness)/len(self.population):.2f}")
            selected_indices = self.selection(amount = 4)
            child1, child2 = self.crossover_onepoint(self.population[selected_indices[0]], self.population[selected_indices[1]])
            child3, child4 = self.crossover_onepoint(self.population[selected_indices[2]], self.population[selected_indices[3]])
            child1, child4 = self.mutation_power(child1), self.mutation_twopoints(child4)
            child_best = self.mutation_power(self.best_indiv)
            #t1 = time.time()
            #new_pop, _ = self.evaluate_population_tournament_parallel(len_selected = 2)
            #t2 = time.time()
            #print(f"Tournament time: {t2-t1}")
            #t1 = time.time()
            new_pop, _ = self.evaluate_population_tournament_from_existing_fitness(len_selected = tournament_size)
            #t2 = time.time()
            #print(f"Tournament from existing fitness time: {t2-t1}")
            new_pop = new_pop + [child1, child2, child3, child4, child_best] + [self.best_indiv] + [self.population[idx] for idx in self.selection()]
            self.population = new_pop + self.create_population(pop_size-len(new_pop))



    def plot_evolution():
        pass


if __name__ == "__main__":
    AGENTS = [ra, aha, apa, apja, cza, ca, ea, paaa, sa, ta]
    IND_SIZE = len(AGENTS)
    ga = GA(IND_SIZE, AGENTS, rounds = 20, tournament_size = 2)
    ga()
    print(ga.best_indiv)