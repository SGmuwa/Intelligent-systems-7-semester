#!/usr/bin/python3
# coding: utf-8
#    Примеры программ для курса "Программирование глубоких нейронных сетей на Python"
#    Copyright (C) 2019  Andrey Sozykin (sozykin@gmail.com)
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
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense, GaussianNoise
from tensorflow.python.keras.datasets import cifar10
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.models import load_model
from tensorflow.keras import utils
from shutil import copyfile
import numpy as np
import datetime
import sys

# Размеры изображения
img_width, img_height = 32, 32
# Размерность тензора на основе изображения для входных данных в нейронную сеть
# backend Tensorflow, channels_last
input_shape = (img_width, img_height, 3)
# Количество эпох
epochs = 10
# Количество изображений для проверки в процентах
validation_split = 0

# Dataset of 50,000 32x32 color training images, labeled over 10 categories, and 10,000 test images.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train / 255.
x_test = x_test / 255.
y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)
classes = ["самолёт", "автомобиль", "птица", "кот", "олень", "пёс", "лягушка", "конь", "корабль", "грузовик"]

import os
if '-l' not in sys.argv:
 if not os.path.exists('my_dense.h5'):
  model = Sequential()
  
  model.add(GaussianNoise(0.01, input_shape=input_shape))
  model.add(Conv2D(32, (5, 5)))
  model.add(Activation('relu'))
  
  model.add(GaussianNoise(0.01))
  model.add(Conv2D(16, (3, 3)))
  model.add(Activation('relu'))
  
  model.add(Flatten())
  model.add(Dropout(0.25)) # Исключить переобучение
  model.add(Dense(10))
  model.add(Activation('sigmoid'))
  
  model.compile(loss='categorical_crossentropy',
                optimizer=Adagrad(lr=1e-3, epsilon=1e-10),
                metrics=['accuracy'])
 else:
  model = load_model('my_dense.h5')
 print(model.summary())
 history_scores = []
 i = 0
 print('Start learning...')
 while True:
  model.fit(x_train, y_train, epochs=epochs, validation_split=validation_split, verbose=0)
  history_scores.append(model.evaluate(x_test, y_test, verbose=0))
  print(datetime.datetime.now(), 'epoch: ', i, 'loss: ', history_scores[-1][0], 'accurate: ', history_scores[-1][1])
  if history_scores[-1][0] > history_scores[0][0]:
   break
  if len(history_scores) > 10:
   del history_scores[0]
  model.save('my_dense.h5')
  copyfile('my_dense.h5', 'my_dense_backup.h5')
  i = i + 1
else:
 from tensorflow.keras.models import load_model
 model = load_model('my_dense.h5')

predictions = model.predict(x_train)
print('predictions[0]:', predictions[0])
print('np.argmax(predictions[0])', np.argmax(predictions[0]))
print('np.argmax(y_train[0])', np.argmax(y_train[0]))
scores = model.evaluate(x_test, y_test, verbose=1)
print('loss: ', scores[0], 'accurate: ', scores[1])

from img_recognizer import recognize_img
recognize_img(model, classes, input_shape, 'rgb')
