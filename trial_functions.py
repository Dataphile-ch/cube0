#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 18:36:05 2025

@author: william
"""

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
