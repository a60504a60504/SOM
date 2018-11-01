#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 20:12:14 2018

@author: imslab
"""
import matplotlib.pylab as plt
import numpy as np
import pickle
from minisom import MiniSom

MapSize = 90
Feature = 4
Sigma = 0.99
LearningRate = 0.5
Iteration = 3000
NeighborVariance = 0.01



if __name__ == "__main__":    
    som = MiniSom(MapSize, MapSize, Feature, sigma=Sigma, learning_rate=LearningRate)
    atk = np.zeros((MapSize, MapSize), dtype=np.int)
    with open('som.p', 'rb') as infile:
        som._weights = pickle.load(infile)
    with open('DumpData.data', 'rb') as infile:
        DumpData = pickle.load(infile) 
    with open('DumpTestData.data', 'rb') as infile:
        DumpTestData = pickle.load(infile)
    """
    with open('AtkMap.data', 'rb') as infile:
        atk = pickle.load(infile)
    """
    
    atk = som.distance_map()
    for Col in range(MapSize):
        for Row in range(MapSize):
                atk[Col,Row] = 1 if atk[Col,Row] > NeighborVariance else 0
    
    plt.figure(figsize=(MapSize,MapSize))
    plt.pcolor(som.distance_map().T, cmap='bone_r')
    for Col in range(MapSize):
        for Row in range(MapSize):
            if (atk[Col,Row]==1):
                plt.plot(Col+.5, Row+.5, 'd', markerfacecolor='None', markeredgecolor='C1', markersize=20, markeredgewidth=8)
    plt.axis([0, MapSize, 0, MapSize])
    plt.savefig('/home/imslab/som_TCPdump.png')
    plt.show()
    
    
    atkCnt = 0
    NonatkCnt = 0
    DR = 0
    FP = 0
    t = DumpTestData['atk']
    for cnt, xx in enumerate(DumpTestData[['port_in','port_out','srcip','desip']].values.tolist()):
        w = som.winner(xx)  # getting the winner
        if (t[cnt]==1):
            atkCnt = atkCnt + 1
            if (atk[w]==1): DR = DR + 1
        else:
            NonatkCnt = NonatkCnt + 1
            if (atk[w]==1): FP = FP + 1
    DR = float(DR)/atkCnt
    FP = float(FP)/NonatkCnt
    print DR
    print FP