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

n = 300 # number of samples
k = 12 # sample depth
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
first_diff = np.insert(first_diff, 0, 0, axis=1)

second_diff = np.zeros((n,k-2))
for i,r in enumerate(results) :
    second_diff[i] = r[2:] - r[:-2]
second_diff = np.insert(second_diff, 0, 0, axis=1)
second_diff = np.insert(second_diff, 0, 0, axis=1)

third_diff = np.zeros((n,k-3))
for i,r in enumerate(results) :
    third_diff[i] = r[3:] - r[:-3]
third_diff = np.insert(third_diff, 0, 0, axis=1)
third_diff = np.insert(third_diff, 0, 0, axis=1)
third_diff = np.insert(third_diff, 0, 0, axis=1)

#%%Plots

fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, layout='constrained')

colours = [row[0] for row in results ]

x = [x for x in range(1,k+1) ]

for r in range(n) :
    y= results[r]
    axs[0,0].scatter(x, y)  # , c=colours)
    axs[0,0].set_ylabel('Alignment Entropy', size=8)

for r in range(n) :
    y= first_diff[r]
    axs[0,1].scatter(x, y)  # , c=colours)
    axs[0,1].set_ylabel('First Difference', size=8)
    axs[0,1].axhline(0)
    perc = (first_diff > 0).sum() / first_diff.size
    axs[0,1].text(0.75, 0.9, f'{perc:.0%} > 0', size=8, transform=axs[0,1].transAxes)

for r in range(n) :
    y= second_diff[r]
    axs[1,0].scatter(x, y)  # , c=colours)
    axs[1,0].set_ylabel('Second Difference', size=8)
    axs[1,0].axhline(0)
    perc = (second_diff > 0).sum() / second_diff.size
    axs[1,0].text(0.75, 0.9, f'{perc:.0%} > 0', size=8, transform=axs[1,0].transAxes)

for r in range(n) :
    y= third_diff[r]
    axs[1,1].scatter(x, y)  # , c=colours)
    axs[1,1].set_ylabel('Third Difference', size=8)
    axs[1,1].axhline(0)
    perc = (third_diff > 0).sum() / third_diff.size
    axs[1,1].text(0.75, 0.9, f'{perc:.0%} > 0', size=8, transform=axs[1,1].transAxes)

fig.suptitle('Comparison of Rewards per Depth of Search', size=10)

plt.show()