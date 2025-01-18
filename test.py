# -*- coding: utf-8 -*-

from cube import Cube
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

mycube = Cube()
mycube.reset()


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

### Main

def vector_cube(cube) :
    """ returns some kind of vector representation of cube
    with each row = a face and each col = colour 1 or 0
    """
    flat = np.reshape(cube, (-1))
    out = np.zeros((len(flat),max(flat)+1))
    for i, f in enumerate(flat) :
        out[i, f] = 1
    return out

def matrix_dist(cube) :    
    cube0=np.zeros((6,3,3), dtype=int)
    for f in range(6):
        cube0[f] = f
    cube0 = vector_cube(cube0)
    cube1 = vector_cube(cube)
    dist = np.linalg.norm(cube1-cube0)
    return dist

def tensor_dist(cube) :
    cube0=np.zeros((6,3,3), dtype=int)
    for f in range(6):
        cube0[f] = f
    dist = tf.norm(tf.convert_to_tensor(cube - cube0, dtype=float))
    return dist.numpy()

# Generate n sample cubes at each level l
test_cube = Cube()
results = []
for l in range(16+1) :
    for n in range(20) :
        test_cube.cube = sample_cube(l)
        test_cube.update_entropy('naive')
        naive=test_cube.entropy
        test_cube.update_entropy('align')
        align=test_cube.entropy
        test_cube.update_entropy('nn')
        nn=test_cube.entropy
        tnorm = matrix_dist(test_cube.cube)
        results.append([l,naive,align, nn, tnorm])

#colours = [row[0] for row in results if row[0] in (2,5)]
#x= [row[3] for row in results if row[0] in (2,5)]
#y= [row[2] for row in results if row[0] in (2,5)]
colours = [row[0] for row in results ]
x= [row[0] for row in results ]
y= [row[3] for row in results ]

plt.scatter(x, y, c=colours)
plt.title('Comparison of Cube Entropy Measures')
plt.xlabel('Actual Entropy')
plt.ylabel('NN Entropy')
plt.xlim(0,16)
plt.ylim(0,16)
plt.show()

x= [row[0] for row in results ]
y= [row[2] for row in results ]

plt.scatter(x, y, c=colours)
plt.title('Comparison of Cube Entropy Measures')
plt.xlabel('Actual Entropy')
plt.ylabel('Align Entropy')
plt.xlim(0,16)
plt.ylim(0,34)
plt.show()

x= [row[0] for row in results ]
y= [row[1] for row in results ]

plt.scatter(x, y, c=colours)
plt.title('Comparison of Cube Entropy Measures')
plt.xlabel('Actual Entropy')
plt.ylabel('Naive Entropy')
plt.xlim(0,16)
plt.ylim(0,54)
plt.show()

x= [row[0] for row in results ]
y= [row[4] for row in results ]

plt.scatter(x, y, c=colours)
plt.title('Comparison of Cube Entropy Measures')
plt.xlabel('Actual Entropy')
plt.ylabel('Euclidian Entropy')
plt.xlim(0,16)
plt.ylim(0,15)
plt.show()

"""
Conclusions: 
    Neither the align nor the naive entropy distinguish between cubes more than 5-6 moves from solved state.
    Align seems to have a little more differentiation.
    Theoretically good measures, such as vector or tensor norms, 
    do not do as well as the "informed" align measure.
"""

    