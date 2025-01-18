# -*- coding: utf-8 -*-

import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np
from sklearn.model_selection import train_test_split
from cube import Cube
from copy import deepcopy
import pickle


# Step 1 - generate sample cubes for training.
# cubes at k=0 : 1
# cubes at k=1 : 18
# cubes at k=2 : 324
# cubes at k=3 : 5832
# cubes at k=4 : 104,976
# cubes at k=5 : 1,889,568

def vector_cube(cube) :
    """ returns some kind of vector representation of cube
    with each row = a face and each col = colour 1 or 0
    """
    flat = np.reshape(cube, (-1))
    out = np.zeros((len(flat),max(flat)+1))
    for i, f in enumerate(flat) :
        out[i, f] = 1
    return out

def sample_cube(cube, k) :
    """
    Generate 1 sample cube at depth k rotates from solved state.
    Returns np.array 6x3x3

    """
    cube.reset()    
    move = []
    i = 0
    while i < k :
        move.extend(cube.rand_move(3))
        move = cube.compress_moves( move )
        i = len(move)
    move = move[:k]
    cube.move(move)

    return k, cube.cube

def make_samples(cube, n_samples=10, k_depth=3) :
        X = []
        Y = []
        for k in range(k_depth+1) :
            print(f"Generating at Level : {k}")
            for i in range(n_samples) :
            # generate n_samples of cubes
                y, A = sample_cube(cube, k)
                vector_A = vector_cube(A)
                X.append(deepcopy(vector_A))
                Y.append(y)

        return np.array(X), np.array(Y)

### Main

max_depth = 6
gen_samples = False
if gen_samples :
    cube = Cube()
    X, Y = make_samples(cube, n_samples=100000, k_depth=max_depth)
    with open("X_samples.pkl", "wb") as f :
        pickle.dump(X, f)
    with open("Y_samples.pkl", "wb") as f :
        pickle.dump(Y, f)
else :
    with open("X_samples.pkl", "rb") as f :
        X = pickle.load(f)
    with open("Y_samples.pkl", "rb") as f :
        Y = pickle.load(f)


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, random_state=42)

#inputs = keras.Input(shape=(6,3,3), name="cubes")
#hidden_1 = layers.Flatten()(inputs)
inputs = keras.Input(shape=(54,6), name="cubes")
hidden_2 = layers.Dense(8192, activation="relu", name="dense_1")(inputs)
hidden_3 = layers.Dense(1024, activation="relu", name="dense_2")(hidden_2)
hidden_4 = layers.Dense(512, activation="relu", name="dense_3")(hidden_3)
outputs = layers.Dense(max_depth+1, activation="softmax", name="predictions")(hidden_4)

model = keras.Model(inputs=inputs, outputs=outputs)

"""
model.compile(
    optimizer=keras.optimizers.RMSprop(),  # Optimizer
    # Loss function to minimize
    loss=keras.losses.MeanSquaredError(),
    # List of metrics to monitor
    metrics=[keras.metrics.Precision()],
)
"""

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.RMSprop(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

print("Fit model on training data")
history = model.fit(
    X_train,
    y_train,
    batch_size=64,
    epochs=10,
    shuffle=True,
    validation_split=0.2,
)

eval_scores = model.evaluate(X_test, y_test, verbose=2)
print("Eval loss:", eval_scores[0])
print("Eval accuracy:", eval_scores[1])

model.save("nn_entropy.keras")

