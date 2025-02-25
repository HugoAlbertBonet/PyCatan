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

from Managers.GameDirector import GameDirector
AGENTS = [ra, aha, apa, apja, cza, ca, ea, paaa, sa, ta]

IND_SIZE = len(AGENTS)




def game(all_agents):
    # Ejemplo de ejecución
    try:
        game_director = GameDirector(agents=all_agents, max_rounds=200, store_trace=False)
        game_trace = game_director.game_start(print_outcome=False)
        return game_trace
    except Exception as e:
        print(f"Error: {e}")
        return 0


def extract_winner(all_agents, chosen_agent, game_trace):
    # Análisis de resultados
    last_round = max(game_trace["game"].keys(), key=lambda r: int(r.split("_")[-1]))
    last_turn = max(game_trace["game"][last_round].keys(), key=lambda t: int(t.split("_")[-1].lstrip("P")))
    victory_points = game_trace["game"][last_round][last_turn]["end_turn"]["victory_points"]
    
    winner = max(victory_points, key=lambda player: int(victory_points[player]))
    fitness = 0
    if all_agents.index(chosen_agent) == int(winner.lstrip("J")):  
        fitness += 1

    return fitness, int(winner.lstrip("J"))

def create_indiv():
    a = [random.random() for i in range(IND_SIZE)]
    suma = sum(x for x in a)
    a = [x/suma for x in a]
    return a
        

trace = game([ra, aha, apa, apja])
fit = extract_winner([apa, apa, apa, apa], apa, trace)
print(fit)


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.random)

ind1 = creator.Individual(create_indiv())

print(ind1)               
print(ind1.fitness.valid) 

def evaluate(individual):
    player = np.random.choice(AGENTS, size = 1, p = individual)
    others = np.random.choice(AGENTS, size=3)
    print(f"Game: {others} vs {player}")
    idx = random.randint(0,3)
    order = list(others)[:idx]+list(player)+list(others)[idx:]
    print(idx, order)
    trace = game(order)
    a = extract_winner(order, player[0], trace)
    return a

print(evaluate(ind1))

#ind1.fitness.values = evaluate(ind1)
#print(ind1.fitness.valid)    
#print(ind1.fitness)          