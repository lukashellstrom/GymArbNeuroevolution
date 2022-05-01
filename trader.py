import csv
from random import randint
"""
Utför köp och säljordrar med de virituella pengarna.
"""

usd_bal = 1000
usd_init = usd_bal
cr_bal = 0.0
announce_actions = False

holds = 0
buys = 0
sells = 0

def buy(usd_am, cr_price, log, rang):
    global usd_bal, cr_bal, buys
    cr_am = usd_am/cr_price
    if usd_bal > usd_am:
        usd_bal -= usd_am
        cr_bal += cr_am*1
        buys += 1
        if log is not None:
            log.writerow({'type':'buy', 'cr_am':cr_am, 'usd_am':usd_am, 'cr_price':cr_price, 'time':rang})
        if announce_actions:
            print('Bought', cr_am, 'CR for €' + str(usd_am) + f' (CR-EUR: {cr_price})' + '\n' + 'CR Balance:', str(cr_bal), 'CR (Worth €' + str(round(cr_price*cr_bal, 2)) + '), USD Balance: €' + str(usd_bal) + '\n')

    else:
        pass
        #print('Insufficent funds')

def sell(usd_am, cr_price, log, rang):
    global usd_bal, cr_bal, sells
    cr_am = usd_am/cr_price
    if cr_bal > cr_am:
        cr_bal -= cr_am
        usd_bal += usd_am*1
        sells += 1
        if log is not None:
            log.writerow({'type':'sell', 'cr_am':cr_am, 'usd_am':usd_am, 'cr_price': cr_price, 'time':rang})
        if announce_actions: 
            print('Sold', cr_am, 'CR for €' + str(usd_am) + f' (CR-EUR: {cr_price})' + '\n' + 'CR Balance:', str(cr_bal), 'CR (Worth €' + str(round(cr_price*cr_bal, 2)) + '), USD Balance: €' + str(usd_bal) + '\n')
    else:
        pass
        #print('Insufficent funds')

def act_prediction(probs, cr_pric, log, rang):
    #Kallas efter varje 5min intervall. Utför vald utgärd
    global cr_bal, usd_bal, holds, saves
    if probs.index(max(probs)) == 0 or probs[1] == probs[2]:
        #Hold. Gör ingenting. Används även när köp och sälj är samma sannolikhet
        if announce_actions:
            #print('Hold')
            pass
        holds += 1
        
    elif probs.index(max(probs)) == 1:
        """
        Köper kryptovaluta i förhållande till hur säker datorn är på att det är rätt åtgärd.
        Jag la även till faktorn 0.4 för att undvika att det köps för alla pengar på en gång.
        Jag vet egentligen inte ifall det är nödvändigt men tänkte att det skulle skynda på
        träningen av modellen. Denna faktor bör experimenteras med för att optimera vinstnen
        """
        if usd_bal*probs[1]*0.4 >= 10:
            buy(usd_bal*probs[1]*0.4, cr_pric, log, rang)

    elif probs.index(max(probs)) == 2:
        #Säljer krypto i förhållande till nuvrande innehavet (likt köp ordern)
        if cr_bal*cr_pric*probs[2]*0.4 >= 10:
            sell(cr_bal*cr_pric*probs[2]*0.4, cr_pric, log, rang)