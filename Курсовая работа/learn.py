#!/usr/bin/python3
# coding: utf-8
#    Homework of Intelligent systems in 8 semester.
#    Copyright (C) 2020  Sidorenko Mikhail Pavlovich (motherlode.muwa@gmail.com)
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

from tensorflow.keras.losses import mean_absolute_error
from degrade_data import generator_bad_and_good_sound
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense
from download_data import getFileIterator
import numpy as np

model = Sequential()
model.add(Dense(1000, input_dim=980, activation="sigmoid"))
model.add(Dense(1000, activation="sigmoid"))
model.add(Dense(1000, activation="sigmoid"))
model.compile(
    optimizer = SGD(learning_rate=0.01, momentum=0.0, nesterov=False),
    metrics = ["accuracy"],
    loss = mean_absolute_error)

default_array = np.zeros((980), dtype=float)
X = None
Y = None
for path in getFileIterator():
    for (x, y) in generator_bad_and_good_sound(path):
        if len(x[0]) != 980:
            target_size = np.zeros((980))
            target_size[:x[0].shape[0]] = x[0]
            x = np.array([target_size])
        if len(y[0]) != 1000:
            target_size = np.zeros((1000))
            target_size[:y[0].shape[0]] = y[0]
            y = np.array([target_size])
        if X is None:
            X = x
        else:
            X = np.concatenate((X, x))
        if Y is None:
            Y = y
        else:
            Y = np.concatenate((Y, y))
        print('\r{:.2%} '.format(len(X) / 100000.), end='', flush=True)
        if len(X) >= 100000:
            print()
            model.fit(X, Y, epochs=100, validation_split=0.1, verbose=1)
            X = None
            Y = None
    print()
