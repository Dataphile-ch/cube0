#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 11:41:57 2025

@author: william

Explore whether the entropy difference increases as we move away from the solved state.

"""

#%% Pakcages and Constants
from cube import Cube
import matplotlib.pyplot as plt
import numpy as np

mycube = Cube()
mycube.reset()

#%% Functions



#%% Procedure Section

n = 100 # number of samples
k = 15 # sample depth
method = 'align'

results = np.zeros((n,k))
for i in range(n) :
    mycube.reset()
    move = mycube.rand_move(k)
    ent = []
    for r in move :
        mycube.rotate(r)
        mycube.update_entropy(method)
        ent.append(mycube.entropy)
    results[i] = ent

first_diff = np.zeros((n,k-1))
for i,r in enumerate(results) :
    first_diff[i] = r[1:] - r[:-1]

second_diff = np.zeros((n,k-2))
for i,r in enumerate(results) :
    second_diff[i] = r[2:] - r[:-2]

third_diff = np.zeros((n,k-3))
for i,r in enumerate(results) :
    third_diff[i] = r[3:] - r[:-3]
