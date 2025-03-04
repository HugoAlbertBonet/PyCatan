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

from deap import base
from deap import creator
from deap import tools
import numpy as np
import time
import concurrent.futures

from Managers.GameDirector import GameDirector

class GA():
    def __init__(self, ind_size, agents, rounds = 50):
        self.ind_size = ind_size
        self.agents = agents
        self.rounds = rounds
        self.population = []


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
        self.population = []
        for i in range(pop_size):
            self.population.append(self.create_individual())


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
        return a, idx #a is the order inside the game (0to3), and idx is the order of the individuals in the game (if idx[0]==1 it means that individual 1 had the first turn (turn 0))



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
    

    def evaluate_population_tournament_parallel(self):
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
            indivs_ordered = sorted([(v,k) for k,v in indiv_fitness.items()])
            new_fit, new_ind = indivs_ordered[0]
            new_fitness.append(new_fit/self.rounds)
            selected.append(indivs[new_ind])
        return selected, new_fitness

    def mutation():
        pass

    def crossover():
        pass


    def selection():
        pass

    def replacement():
        pass

    def __call__():
        pass

    def plot_evolution():
        pass


if __name__ == "__main__":
    AGENTS = [ra, aha, apa, apja, cza, ca, ea, paaa, sa, ta]
    IND_SIZE = len(AGENTS)
    ga = GA(IND_SIZE, AGENTS)
    
    ga.create_population(40)
    t1 = time.time()
    a = ga.evaluate_population_tournament_parallel()
    t2 = time.time()
    print(t2-t1)

    t1 = time.time()
    a = ga.evaluate_population_tournament()
    t2 = time.time()
    print(t2-t1)