#!/usr/bin/python3
from tensorflow.keras.datasets import fashion_mnist
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
x_train = x_train / 255
x_test = x_test / 255

y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)
classes = ['футболка', 'брюки', 'свитер', 'платье', 'пальто', 'туфли', 'рубашка', 'кроссовки', 'сумка', 'ботинки']

if '-p' in sys.argv:
 for a, b in zip(x_train, y_train):
  print(str(classes[np.argmax(b)]))
  for i in range(len(a)):
   print(' ' if a[i] < 0.33 else ('-' if a[i] < 0.66 else 'M'), end='')
   if (i + 1) % 28 == 0:
    print()
  print()
 exit()

if '-h' in sys.argv:
 print('learn fashion mnist.\n-h\thelp\n-p\tprint data\n-c\tcreate new model')
 exit()

model = None

if '-c' in sys.argv or not os.path.exists('dense.h5'):
 model = Sequential()
 model.add(Dense(800, input_dim=784, activation="relu"))
 model.add(Dense(10, activation="softmax"))
 model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])
 print(model.summary())
 model.fit(x_train, y_train, batch_size=200, epochs=100, validation_split=0.2, verbose=1)
 model.save('dense.h5')
else:
 model = load_model('dense.h5')

predictions = model.predict(x_train)
print('predictions[0]:', predictions[0])
print('np.argmax(predictions[0])', np.argmax(predictions[0]))
print('np.argmax(y_train[0])', np.argmax(y_train[0]))
print('scores[1]: ', model.evaluate(x_test, y_test, verbose=1)[1])

from img_recognizer import recognize_img
recognize_img(model, classes, (28, 28), "grayscale")
