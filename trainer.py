import multiprocessing
import random
import numpy as np
import sys
from tester import Account
from df_parser import Parser
from test_trader import Currency
from elitism import eaSimpleWithElitism
from tester import backtester
from deap import base
from deap import creator
from datetime import date
from deap import tools
import logging
import pandas as pd
import time as t
from math import floor
from config import CROWDING_FACTOR, POPULATION_SIZE, HALL_OF_FAME_SIZE, P_CROSSOVER, P_MUTATION, MAX_GENERATIONS, \
    STRATEGIES, START, END, CUSHION, PIPS_MULTIPLIER, FILES_DIR
# import os

CPU_COUNT = multiprocessing.cpu_count()-1
STRATEGY = STRATEGIES[str(sys.argv[2])]
pair_name = str(sys.argv[1])
test = False
# start = datetime.fromisoformat(str(sys.argv[2]))
# end = datetime.fromisoformat(str(sys.argv[3]))


def classificationAccuracy(individual):
    pair = Currency(pair=pair_name, strategy=STRATEGY, params=individual, cushion=CUSHION)
    account = Account(1, len(pair.open))

    for n in range(pair.start_pos + 2, len(pair.open)):
        if account.balance > 3 * account.lot_size * 100:
            if account.balance >= 200:
                if round(floor(account.balance) / 10000, 2) > account.lot_size:
                    account.lot_size = round(floor(account.balance) / 10000, 2)

            pair.tester(
                account=account,
                index=n,
            )

            if account.balance > account.balance_max:
                account.balance_max = account.balance

            if account.balance < account.balance_min:
                account.balance_min = account.balance

            if 100 / account.balance_max * account.balance < account.drawdown:
                account.drawdown = 100 / account.balance_max * account.balance
        else:
            break

    result = account.drawdown + account.pips * PIPS_MULTIPLIER

    # if account.drawdown < 10:
    #     result -= 1000000

    # if pair.profit_longs < pair.strategy_dict["total_trades"] / 5:
    #     result -= 5000

    # if pair.profit_shorts < pair.strategy_dict["total_trades"] / 5:
    #     result -= 5000

    if pair.total_trades < pair.strategy["total_trades"]:
        result -= 10000

    return result,


toolbox = base.Toolbox()
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

for i in range(len(STRATEGY["bounds_high"])):
    toolbox.register("hyper_parameter_" + str(i), random.uniform, STRATEGY["bounds_low"][i],
                     STRATEGY["bounds_high"][i])

hyper_parameters = ()
for i in range(len(STRATEGY["bounds_high"])):
    hyper_parameters = hyper_parameters + (toolbox.__getattribute__("hyper_parameter_" + str(i)),)

toolbox.register("individualCreator", tools.initCycle, creator.Individual, hyper_parameters, n=1)
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

toolbox.register("evaluate", classificationAccuracy)
toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=STRATEGY["bounds_low"],
                 up=STRATEGY["bounds_high"], eta=CROWDING_FACTOR)
toolbox.register("mutate", tools.mutPolynomialBounded, low=STRATEGY["bounds_low"],
                 up=STRATEGY["bounds_high"], eta=CROWDING_FACTOR,
                 indpb=1.0 / len(STRATEGY["bounds_high"]))


if __name__ == "__main__":
    print("Preparing Market Data...")
    # Prepare Market Data
    start = t.time()
    Parser(pair=pair_name, strategy=STRATEGY, start=START, end=END, save=True)
    end = t.time()
    print("Elapsed (with compilation) = %s" % (end - start))

    print("Preparing Numba...")
    # Prepare Numba
    start = t.time()
    Currency(pair=pair_name, strategy=STRATEGY, params=STRATEGY["bounds_low"])
    end = t.time()
    print("Elapsed (with compilation) = %s" % (end - start))

    print("Starting the Algorithm...")

    pool = multiprocessing.Pool(processes=CPU_COUNT)
    toolbox.register("map", pool.map)

    population = toolbox.populationCreator(n=POPULATION_SIZE)
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    stats.register("avg", np.mean)

    population, logbook = eaSimpleWithElitism(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                              ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=True)

    pool.close()

    params = hof.items[0]

    print(params)

    test_results = backtester(pairs=[pair_name], strategy=STRATEGY, params=[params])

    score = round(hof.items[0].fitness.values[0], 2)

    if not test:
        params_data = pd.read_csv(FILES_DIR + "params_data.csv", index_col=0)
        params_data[pair_name]['params'] = params
        params_data[pair_name]['update_date'] = date.today()
        params_data[pair_name]['winning_rate'] = test_results[0]
        params_data[pair_name]['max_drawdown'] = test_results[1]
        params_data[pair_name]['average_profit'] = test_results[2]
        params_data[pair_name]['average_loss'] = test_results[3]
        params_data[pair_name]['total_trades'] = test_results[4]
        params_data[pair_name]['pips'] = round(test_results[5])
        params_data[pair_name]['balance'] = test_results[6]
        params_data[pair_name]['fitness'] = score
        params_data[pair_name]['testing_from'] = START.date()
        params_data[pair_name]['testing_to'] = END.date()

        new_params_df = pd.DataFrame(data=params_data)
        new_params_df.to_csv(FILES_DIR + "params_data.csv")

        logging.basicConfig(filename=FILES_DIR + "the_bot.log", format='%(asctime)s %(message)s', filemode='a')
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        logger.info(f"New {pair_name} parameters: {params}, winning rate: {test_results[0]}%, "
                    f"max drawdown: {test_results[1]}%, average profit: ${test_results[2]}, "
                    f"average loss: ${test_results[3]}, total trades: {test_results[4]}, "
                    f"pips: {round(test_results[5], 2)}, balance: {test_results[6]}, fitness: {score}")

        # exec(open(str(sys.argv[2])).read())
    else:
        print("- Best solution is: ")
        print("params = ", hof.items[0])
        print("Accuracy = %1.5f" % hof.items[0].fitness.values[0])
        print(f"New {pair_name} parameters: {params}, winning rate: {test_results[0]}%, "
              f"max drawdown: {test_results[1]}%, average profit: ${test_results[2]}, "
              f"average loss: ${test_results[3]}, total trades: {test_results[4]}, "
              f"pips: {round(test_results[5], 2)}, balance: {test_results[6]}, fitness: {score}")
