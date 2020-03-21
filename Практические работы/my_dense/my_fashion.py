#!/usr/bin/python3
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
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import utils
import numpy as np
import sys
import os

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train / 255 * 2 - 1
x_test = x_test / 255 * 2 - 1

y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)
classes = ['футболка', 'брюки', 'свитер', 'платье', 'пальто', 'туфли', 'рубашка', 'кроссовки', 'сумка', 'ботинки']

if '-p' in sys.argv:
 for a, b in zip(x_train, y_train):
  print(str(classes[np.argmax(b)]))
  for i in range(len(a)):
   print(' ' if a[i] < -0.33 else ('-' if a[i] < 0.33 else 'M'), end='')
   if (i + 1) % 28 == 0:
    print()
  print()
 exit()

if '-h' in sys.argv:
 print('learn fashion mnist.\n-h\thelp\n-p\tprint data\n-c\tcreate new model\n-l\tload model.')
 exit()

model = None

if '-c' in sys.argv or '-l' not in sys.argv:
 if os.path.exists('my_dense.h5') and os.path.exists('my_dense_backup.h5'):
  if os.path.getsize('my_dense.h5') >= os.path.getsize('my_dense_backup.h5'):
   model = load_model('my_dense.h5')
  else:
   model = load_model('my_dense_backup.h5')
 elif os.path.exists('my_dense.h5'):
  model = load_model('my_dense.h5')
 elif os.path.exists('my_dense_backup.h5'):
  model = load_model('my_dense_backup.h5')
 else:
  model = Sequential()
  model.add(Dense(784, input_dim=784, activation="sigmoid"))
  model.add(Dense(784, activation="sigmoid"))
  model.add(Dense(784, activation="relu"))
  model.add(Dense(10, activation="softmax"))
  model.compile(
   loss="categorical_crossentropy",
   optimizer=Adagrad(lr=1e-3, epsilon=1e-10),
   metrics=["accuracy"])
 print(model.summary())
 history_scores = []
 i = 0
 print('Start learning...')
 while True:
  model.fit(x_train, y_train, batch_size=60000, epochs=100, validation_split=0.2, verbose=0)
  history_scores.append(model.evaluate(x_test, y_test, verbose=0))
  if history_scores[-1][0] > history_scores[0][0]:
   break
  print('epoch: ', i, 'loss: ', history_scores[-1][0], 'accurate: ', history_scores[-1][1])
  if len(history_scores) > 10:
   del history_scores[0]
  model.save('my_dense.h5')
  from shutil import copyfile
  copyfile('my_dense.h5', 'my_dense_backup.h5')
  i = i + 1
else:
 model = load_model('my_dense.h5')

predictions = model.predict(x_train)
print('predictions[0]:', predictions[0])
print('np.argmax(predictions[0])', np.argmax(predictions[0]))
print('np.argmax(y_train[0])', np.argmax(y_train[0]))
print('scores[1]: ', model.evaluate(x_test, y_test, verbose=1)[1])

from img_recognizer import recognize_img
recognize_img(model, classes, (28, 28), "grayscale")
