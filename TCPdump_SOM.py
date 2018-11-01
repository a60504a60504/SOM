#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 11:08:45 2018

@author: imslab
"""
import pandas as pd
import numpy as np
import pickle
from minisom import MiniSom

MapSize = 90
Feature = 4
Sigma = 0.99
LearningRate = 0.5
Iteration = 3000

"""
def ln_decay(learning_rate, t, max_iter):
    return learning_rate * (1 - t/(max_iter/np.log(learning_rate)))
"""

def IP2Int(ip_series):
    o = ip_series.str.split('.',expand=True).astype(int)
    res = (16777216 * o[0]) + (65536 * o[1]) + (256 * o[2]) + o[3]
    return res

def OpenTrainData():
    DumpData = pd.read_csv('~/DARPA/darpa/training_data/week_1/monday.list', sep=" ", header=None)
    DumpData_T = pd.read_csv('~/DARPA/darpa/training_data/week_1/tuesday.list', sep=" ", header=None)
    DumpData = DumpData.append(DumpData_T, ignore_index=True)
    DumpData_T = pd.read_csv('~/DARPA/darpa/training_data/week_1/wednesday.list', sep=" ", header=None)
    DumpData = DumpData.append(DumpData_T, ignore_index=True)
    DumpData_T = pd.read_csv('~/DARPA/darpa/training_data/week_1/thursday.list', sep=" ", header=None)
    DumpData = DumpData.append(DumpData_T, ignore_index=True)
    DumpData_T = pd.read_csv('~/DARPA/darpa/training_data/week_1/friday.list', sep=" ", header=None)
    DumpData = DumpData.append(DumpData_T, ignore_index=True)
    DumpData_T = pd.read_csv('~/DARPA/darpa/training_data/week_2/monday.list', sep=" ", header=None)
    DumpData = DumpData.append(DumpData_T, ignore_index=True)
    DumpData_T = pd.read_csv('~/DARPA/darpa/training_data/week_2/tuesday.list', sep=" ", header=None)
    DumpData = DumpData.append(DumpData_T, ignore_index=True)
    DumpData_T = pd.read_csv('~/DARPA/darpa/training_data/week_2/wednesday.list', sep=" ", header=None)
    DumpData = DumpData.append(DumpData_T, ignore_index=True)
    DumpData_T = pd.read_csv('~/DARPA/darpa/training_data/week_2/thursday.list', sep=" ", header=None)
    DumpData = DumpData.append(DumpData_T, ignore_index=True)
    DumpData_T = pd.read_csv('~/DARPA/darpa/training_data/week_2/friday.list', sep=" ", header=None)
    DumpData = DumpData.append(DumpData_T, ignore_index=True)
    DumpData_T = pd.read_csv('~/DARPA/darpa/training_data/week_3/monday.list', sep=" ", header=None)
    DumpData = DumpData.append(DumpData_T, ignore_index=True)
    DumpData_T = pd.read_csv('~/DARPA/darpa/training_data/week_3/tuesday.list', sep=" ", header=None)
    DumpData = DumpData.append(DumpData_T, ignore_index=True)
    DumpData_T = pd.read_csv('~/DARPA/darpa/training_data/week_3/wednesday.list', sep=" ", header=None)
    DumpData = DumpData.append(DumpData_T, ignore_index=True)
    DumpData_T = pd.read_csv('~/DARPA/darpa/training_data/week_3/thursday.list', sep=" ", header=None)
    DumpData = DumpData.append(DumpData_T, ignore_index=True)
    DumpData_T = pd.read_csv('~/DARPA/darpa/training_data/week_3/friday.list', sep=" ", header=None)
    DumpData = DumpData.append(DumpData_T, ignore_index=True)
    DumpData_T = pd.read_csv('~/DARPA/darpa/training_data/week_4/monday.list', sep=" ", header=None)
    DumpData = DumpData.append(DumpData_T, ignore_index=True)
    DumpData_T = pd.read_csv('~/DARPA/darpa/training_data/week_4/tuesday.list', sep=" ", header=None)
    DumpData = DumpData.append(DumpData_T, ignore_index=True)
    DumpData_T = pd.read_csv('~/DARPA/darpa/training_data/week_4/wednesday.list', sep=" ", header=None)
    DumpData = DumpData.append(DumpData_T, ignore_index=True)
    DumpData_T = pd.read_csv('~/DARPA/darpa/training_data/week_4/thursday.list', sep=" ", header=None)
    DumpData = DumpData.append(DumpData_T, ignore_index=True)
    DumpData_T = pd.read_csv('~/DARPA/darpa/training_data/week_4/friday.list', sep=" ", header=None)
    DumpData = DumpData.append(DumpData_T, ignore_index=True)
    DumpData_T = pd.read_csv('~/DARPA/darpa/training_data/week_5/monday.list', sep=" ", header=None)
    DumpData = DumpData.append(DumpData_T, ignore_index=True)
    DumpData_T = pd.read_csv('~/DARPA/darpa/training_data/week_5/tuesday.list', sep=" ", header=None)
    DumpData = DumpData.append(DumpData_T, ignore_index=True)
    DumpData_T = pd.read_csv('~/DARPA/darpa/training_data/week_5/wednesday.list', sep=" ", header=None)
    DumpData = DumpData.append(DumpData_T, ignore_index=True)
    DumpData_T = pd.read_csv('~/DARPA/darpa/training_data/week_5/thursday.list', sep=" ", header=None)
    DumpData = DumpData.append(DumpData_T, ignore_index=True)
    DumpData_T = pd.read_csv('~/DARPA/darpa/training_data/week_5/friday.list', sep=" ", header=None)
    DumpData = DumpData.append(DumpData_T, ignore_index=True)
    DumpData_T = pd.read_csv('~/DARPA/darpa/training_data/week_6/monday.list', sep=" ", header=None)
    DumpData = DumpData.append(DumpData_T, ignore_index=True)
    DumpData_T = pd.read_csv('~/DARPA/darpa/training_data/week_6/tuesday.list', sep=" ", header=None)
    DumpData = DumpData.append(DumpData_T, ignore_index=True)
    DumpData_T = pd.read_csv('~/DARPA/darpa/training_data/week_6/wednesday.list', sep=" ", header=None)
    DumpData = DumpData.append(DumpData_T, ignore_index=True)
    DumpData_T = pd.read_csv('~/DARPA/darpa/training_data/week_6/thursday.list', sep=" ", header=None)
    DumpData = DumpData.append(DumpData_T, ignore_index=True)
    DumpData_T = pd.read_csv('~/DARPA/darpa/training_data/week_6/friday.list', sep=" ", header=None)
    DumpData = DumpData.append(DumpData_T, ignore_index=True)
    return DumpData

def OpenTestData():
    DumpData = pd.read_csv('~/DARPA/darpa/training_data/week_7/monday.list', sep=" ", header=None)
    DumpData_T = pd.read_csv('~/DARPA/darpa/training_data/week_7/tuesday.list', sep=" ", header=None)
    DumpData = DumpData.append(DumpData_T, ignore_index=True)
    DumpData_T = pd.read_csv('~/DARPA/darpa/training_data/week_7/wednesday.list', sep=" ", header=None)
    DumpData = DumpData.append(DumpData_T, ignore_index=True)
    DumpData_T = pd.read_csv('~/DARPA/darpa/training_data/week_7/thursday.list', sep=" ", header=None)
    DumpData = DumpData.append(DumpData_T, ignore_index=True)
    DumpData_T = pd.read_csv('~/DARPA/darpa/training_data/week_7/friday.list', sep=" ", header=None)
    DumpData = DumpData.append(DumpData_T, ignore_index=True)
    return DumpData

def Data_preprocessing(DumpData):
    DumpData = DumpData[[6,7,8,9,10]]
    DumpData.columns = ['port_in','port_out','srcip','desip','atk']
    """
    #Onehot-encoding 
    onehot_encoding = pd.get_dummies(DumpData['protocol'], prefix = 'protocol')
    DumpData = DumpData.drop('protocol',1)
    DumpData = pd.concat([onehot_encoding, DumpData],axis=1)
    del onehot_encoding
    """
    #Remove '-'
    DumpData = DumpData.replace('-',0)
    #object to int
    DumpData['port_in'] = DumpData['port_in'].astype(str).astype(int)
    DumpData['port_out'] = DumpData['port_out'].astype(str).astype(int)
    DumpData['atk'] = DumpData['atk'].astype(int)
    #IP to int
    DumpData['srcip'] = IP2Int(DumpData['srcip'])
    DumpData['desip'] = IP2Int(DumpData['desip'])
    return DumpData

if __name__ == "__main__":
    DumpData = OpenTrainData()
    DumpData = Data_preprocessing(DumpData)
    DumpTestData = OpenTestData()
    DumpTestData = Data_preprocessing(DumpTestData)
    
    som = MiniSom(MapSize, MapSize, Feature, sigma=Sigma, learning_rate=LearningRate)
    som.train_random(DumpData[['port_in','port_out','srcip','desip']].values.tolist(), Iteration)

    """    
    atk = np.zeros((MapSize, MapSize), dtype=np.int)
    t = DumpData['atk']
    # use different colors and markers for each label
    for cnt, xx in enumerate(DumpData[['port_in','port_out','srcip','desip']].values.tolist()):
        if (t[cnt]==1):
            w = som.winner(xx)  # getting the winner
            # palce a marker on the winning position for the sample xx
            atk[w] = 1
    """

    
    with open('som.p', 'wb') as outfile:
        pickle.dump(som.get_weights(), outfile)
        
    with open('DumpData.data', 'wb') as outfile:
        pickle.dump(DumpData, outfile)
        
    with open('DumpTestData.data', 'wb') as outfile:
        pickle.dump(DumpTestData, outfile)
    
    """
    with open('AtkMap.data', 'wb') as outfile:
        pickle.dump(atk, outfile)
    """