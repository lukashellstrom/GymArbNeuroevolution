import neat
import os
import csv
import trader
import pickle
from neat.math_util import softmax
from datetime import datetime, timedelta
import numpy as np
import time

from random import uniform

"""
Använder den sparade genome för att tradea utifrån data-setten. Modifierad variant av 'ai-advisor.py'.
"""


data_file, model_file = 'ETH–2021-01-01–2021-12-31.csv', 'ETH–2018-01-01–2020-12-31.pkl'

#Number of 5 min intervals
test_interval = 8060

#rang defaults to 1
rang = 1
init, latest, last = 0, 0, 0

totales = 0
months = 0

def eval_genome(genome, config, log):
    global data_file, rang, latest, init, last, totales, months

    with open('Datasets/' + data_file) as data:
        data_list = list(csv.reader(data)) ; del data_list[0]
        
        # $trader.usd_init invested at the start of the timespan. Used for evalutaion
        if init == 0: init = (trader.usd_init/float(data_list[rang][0]))
            
        #Calculates networth in USD at the start of each timespan. Is used to evaluate genome weekly performance
        nw = trader.usd_bal + (trader.cr_bal*float(data_list[rang][0]))

        #Every genome is tested against one week worth of data
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        latest = float(data_list[rang+test_interval][0])

        for i in range(rang, rang+test_interval):     
            #Data is getting fed into the nn
            output = [round(float(x),2) for x in net.activate([float(f) for f in data_list[i][1:]])]
            trader.act_prediction(output, float(data_list[i][0]), log, i)

        

    """
    Två saldo visas vid varje veckoslut. Det ena visar hur värdet på portföljen som datorn fått
    förvalta, och det andra portföljvärdet ifall alla pengar hade investerats i början av veckan.
    De visar alltså hur väl datorn har förvaltat pengarna.
    """
    print('Saldo (tradeat)', (trader.usd_bal + (trader.cr_bal*float(data_list[rang+test_interval][0])))) ; print('Saldo (orörda)', (nw/float(data_list[rang][0]))*float(data_list[rang+test_interval][0]))
    factor = (trader.usd_bal + (trader.cr_bal*float(data_list[rang+test_interval][0]))) / ((nw/float(data_list[rang][0]))*float(data_list[rang+test_interval][0]))

    #Networth at the end of the interval
    last = (trader.usd_bal + (trader.cr_bal*float(data_list[rang+test_interval][0])))

    #Making sure trading continues at the right point in time
    rang += test_interval

    #How much better the trading bot did compared to untouched money
    totales += factor
    months += 1
    print('Performance against market:', str(round(factor, 3)) + '\n')
    #trader.usd_bal = 1000.0
 
def min_max_scale(subj):
    scaled_subj = []
    for col_num in range(len(subj[0])):
        col_max, col_min = max(subj[:,col_num]), min(subj[:,col_num])
        for k in subj[:,col_num]:
            scaled_subj.append(2*(k - col_min)/(col_max - col_min) - 1)
    return tuple(scaled_subj)

def run(config_path):
    global model_file
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    #Loads the saved genome
    with open(model_file, "rb") as f:
        genome = pickle.load(f)
    
    #Repeats for 52 weeks
    with open('log.csv', 'w+') as log:
        (writer := csv.DictWriter(log, fieldnames=['type', 'cr_am', 'usd_am', 'cr_price', 'time'])).writeheader()
        for i in range(50):
            try:
                eval_genome(genome, config, writer)
            except IndexError:
                print('*** Dataset ended ***')
                break

    #Jämför det slutgiltiga portöljvärdet med marknaden
    print("Market outperformed by a factor of:", last/(latest*init)) ; print(f'Buys: {trader.buys}. Sells: {trader.sells}. Holds: {trader.holds}')
    print(f'Average performance: {totales/months}')
if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_p = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_p)