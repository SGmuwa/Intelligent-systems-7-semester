#!/usr/bin/python3
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import utils
import numpy as np
import sys

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = np.copy(x_train.reshape(60000, 784))
x_train = x_train / 255

y_train = utils.to_categorical(y_train, 10)
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
 print('learn fashion mnist.\n-h\thelp\n-p\tprint data')
 exit()

model = Sequential()
model.add(Dense(800, input_dim=784, activation="relu"))
model.add(Dense(10, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])
print(model.summary())

model.fit(x_train, y_train, batch_size=200, epochs=100, validation_split=0.2, verbose=1)
model.save('fashion_mnist_dense.h5')

predictions = model.predict(x_train)
print(predictions[0])
print(np.argmax(predictions[0]))
print(np_argmax(y_train[0]))

