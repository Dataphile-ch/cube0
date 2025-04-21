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


def make_samples( n_samples=10, k_depth=3) :
    # NB samples are converted to vector representation of cube.  i.e. flattened.
        cube = Cube()
        X = []
        Y = []
        for k in range(k_depth) :
            print(f"Generating at Level : {k}")
            for i in range(n_samples) :
            # generate n_samples of cubes
                M = cube.rand_move(k)
                cube.move(M)
                vector_A = cube.vector_cube()
                X.append(deepcopy(vector_A))
                Y.append(k)

        return np.array(X), np.array(Y)

#%% Make Sample Data

max_depth = 10
gen_samples = False
if gen_samples :
    X, Y = make_samples( n_samples=10000, k_depth=max_depth)
    with open("X_samples.pkl", "wb") as f :
        pickle.dump(X, f)
    with open("Y_samples.pkl", "wb") as f :
        pickle.dump(Y, f)
else :
    with open("X_samples.pkl", "rb") as f :
        X = pickle.load(f)
    with open("Y_samples.pkl", "rb") as f :
        Y = pickle.load(f)

#%% Model

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, random_state=42)
y_train = keras.utils.to_categorical(y_train, max_depth)
y_test = keras.utils.to_categorical(y_test, max_depth)

model = keras.Sequential(
    [
#     keras.Input(shape=(54,1), name="cubes"),
     layers.Dense(8192, activation="relu", name="dense_1"),
     layers.Dense(1024, activation="relu", name="dense_2"),
     layers.Dense(512, activation="relu", name="dense_3"),
     layers.Dense(max_depth, activation="softmax", name="predictions")
     ]
    )

model.compile(
    loss=keras.losses.CategoricalCrossentropy(),
    optimizer=keras.optimizers.SGD(),
    metrics=[keras.metrics.CategoricalAccuracy()],
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

