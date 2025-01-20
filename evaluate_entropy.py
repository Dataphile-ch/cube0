#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 18:36:05 2025

@author: william

Generate sample cubes
Measure different entropy types
Evaluate 

"""

#%% Packages and setup
from cube import Cube
import matplotlib.pyplot as plt
import numpy as np

mycube = Cube()
mycube.reset()

#%% Functions
def sample_cube(k) :
    """
    Generate 1 sample cube at depth k rotates from solved state.
    Returns np.array 6x3x3

    """
    cube = Cube()
    cube.reset()
    
    move = []
    i = 0
    while i < k :
        move.extend(mycube.rand_move(3))
        move = cube.compress_moves( move )
        i = len(move)
    move = move[:k]
    cube.move(move)

    return cube.cube


#%% Main
# Generate n sample cubes at each level l
test_cube = Cube()
results = []
levels = 10
samples = 10
for l in range(1,levels+1) :
    for n in range(samples) :
        test_cube.cube = sample_cube(l)
        test_cube.update_entropy('naive')
        naive=test_cube.entropy
        test_cube.update_entropy('align')
        align=test_cube.entropy
        test_cube.update_entropy('nn')
        nn=test_cube.entropy
        tnorm = test_cube.matrix_dist()
        results.append([l,naive,align, nn, tnorm])

#colours = [row[0] for row in results if row[0] in (2,5)]
#x= [row[3] for row in results if row[0] in (2,5)]
#y= [row[2] for row in results if row[0] in (2,5)]

fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, layout='constrained')

colours = [row[0] for row in results ]

x= [row[0] for row in results ]

y0= [row[3] for row in results ]
axs[0,0].scatter(x, y0, c=colours)
axs[0,0].set_ylabel('NN Entropy')
axs[0,0].set_xlim(0,16)
axs[0,0].set_ylim(0,levels+1)

y1= [row[2] for row in results ]
axs[0,1].scatter(x, y1, c=colours)
axs[0,1].set_ylabel('Align Entropy')
axs[0,1].set_xlim(0,levels+1)
axs[0,1].set_ylim(0,34)
    
y2= [row[1] for row in results ]
axs[1,0].scatter(x, y2, c=colours)
axs[1,0].set_xlabel('Actual Entropy')
axs[1,0].set_ylabel('Naive Entropy')
axs[1,0].set_xlim(0,levels+1)
axs[1,0].set_ylim(0,54)

y3= [row[4] for row in results ]
axs[1,1].scatter(x, y3, c=colours)
axs[1,1].set_xlabel('Actual Entropy')
axs[1,1].set_ylabel('Euclidian Entropy')
axs[1,1].set_xlim(0,levels+1)
axs[1,1].set_ylim(0,15)

fig.suptitle('Comparison of Cube Entropy Measures')

plt.show()

"""
Conclusions: 
    Neither the align nor the naive entropy distinguish between cubes more than 5-6 moves from solved state.
    Align seems to have a little more differentiation.
    Theoretically good measures, such as vector or tensor norms, 
    do not do as well as the "informed" align measure.
"""

    