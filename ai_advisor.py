import neat
import os
import csv
import trader
import pickle
import numpy as np
import pandas as pd
from neat.math_util import softmax
from math import log10

from random import uniform
"""
Beskrivning:
Används för att träna modellen. Använder data-punkerna från specificerat data set
från 'data_mangager.py'. Inställningarna för hur modellen tränas finns i 
'config-feedforward.txt' filen.
"""

data_file = 'ETH–2018-01-01–2020-12-31.csv'

test_interval = 8060
logbase = 30
tot = None

"""
'rang' används för att hålla koll på var i data filen som varje genom ska börja läsa.
Räknar rader i data filen. Kan användas för att börja vid en punkt i filen. Om man t.ex. 
vill börja efter 1 år ändrar man 'rang' till (60/5) * 24 * 365 = 105 120.
"""
rang = 1

def eval_genomes(genomes, config):
    global data_file, rang, tot, test_interval, logbase
    best = []

    with open('Datasets/' + data_file) as data:
        data_list = list(csv.reader(data)) ; del data_list[0]
        #Varje genom testas mot en vecka av data
        for genome_id, genome in genomes:
            
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            for i in range(rang, rang+test_interval):     

                #Ett fem-min interval i taget matas in i neuronnätet
                '''
                data_window = np.array([item for x,item in enumerate(data_list[i - 8060:i]) if x%144 == 0]).astype(dtype=float, copy=False)
                output = softmax([round(x,2) for x in net.activate(min_max_scale(data_window))])
                '''

                output = [round(float(x),2) for x in net.activate([float(f) for f in data_list[i][1:]])]
                    
                """
                Neuronnätet ger en output med tre variablar. Dessa körs genom ett softmax-filter
                vilket gör att summan av dessa blir 1. Dessa tolkas sedan som sannolikheter för att en
                viss åtgärd är den som resulterar i högst fitness, alltså mest pengar. 
                """
                #Åtgärden som neuronnätet väljer utförs med hjälp av trader.py
                trader.act_prediction(output, float(data_list[i][0]), None, i)
            
            """
            Varje genom betygsätts med hjälp av en fitness function utifrån hur väl den utförde
            sin uppgit, d.v.s. hur mycket pengar den tjänade/förlorade. Fitness function är en
            av skriptets delar där jag gissar att det finns störst utrymme för förbättring. Den
            nuvarande funktionen är skillnaden mellan hela portföljens värde (Likvida + investerade)
            som neuronnätet förvaltat och portföljens värde ifall man i början av veckan hade
            investerat alla pengar och sedan inte rört dem. Alltså hur AIn presterar mot marknaden.
            """
        
            if (((trader.usd_bal + (trader.cr_bal*float(data_list[rang+test_interval][0]))) < (1000/float(data_list[rang][0]))*float(data_list[rang+test_interval][0]))):
                genome.fitness = (((trader.usd_bal + (trader.cr_bal*float(data_list[rang+test_interval][0]))) - (1000/float(data_list[rang][0]))*float(data_list[rang+test_interval][0]))) * (log10(trader.buys + trader.sells + 0.001)/log10(logbase) if log10(trader.buys + trader.sells + 0.001)/log10(logbase) > 1 else 10)
            else:
                genome.fitness = (((trader.usd_bal + (trader.cr_bal*float(data_list[rang+test_interval][0]))) - (1000/float(data_list[rang][0]))*float(data_list[rang+test_interval][0]))) * (log10(trader.buys + trader.sells + 0.001)/log10(logbase) if log10(trader.buys + trader.sells + 0.001)/log10(logbase) > 1 else -1)

            best.append((trader.usd_bal + trader.cr_bal*float(data_list[rang+test_interval][0])) / ((1000/float(data_list[rang][0]))*float(data_list[rang+test_interval][0])))

            # Varje genom börjar veckan med USD $1000 tillgängligt för investering. $1000 är arbiträrt valt
            trader.usd_bal, trader.cr_bal, trader.buys, trader.sells = 1000.0, 0.0, 0, 0

    if (rang + 2*test_interval) < len(data_list):
        #Ifall det finns data kvar i filen börjar nästa genom på nästa vecka. Annars loopar den runt till första veckan
        print(f'{(rang + (test_interval + 4))} < {len(data_list)}?')
        rang += test_interval
    else:
        rang = 1
    if best != []:
        print('Performence against market (factor):', str(round(max(best), 2)))

def min_max_scale(subj):
    scaled_subj = []
    for col_num in range(len(subj[0])):
        col_max, col_min = max(subj[:,col_num]), min(subj[:,col_num])
        for k in subj[:,col_num]:
            scaled_subj.append(2*(k - col_min)/(col_max - col_min) - 1)
    return tuple(scaled_subj)

def run(config_path):
    global data_file
    #Konfigurerar neat
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.StatisticsReporter())

    #Första parametern är funktionen som kallas för att träna varje genom. Andra parametern är antalet generationer
    winner = p.run(eval_genomes, 35)

    #Den genome med högst fitness sparas i 'winner'. Nedan sparas det objektet så att det kan användas igen.
    with open(f'{data_file.split(".")[0]}.pkl', 'wb') as f:
        pickle.dump(winner, f)
        f.close()

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)