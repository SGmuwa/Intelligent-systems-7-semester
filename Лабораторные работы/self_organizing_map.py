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
# Кохонен
# 1. Функция близости нейронов: Гауссова функция.
# 2. Полносвязная нейронная сеть.
# 3. Вычисляется нейрон-победитель с помощью евклидово расстояния.
# 4. Используются только нормированные векторы данных.
# 5. Обучение без учителя.
# 6. Используется для кластеризации.

import matplotlib.pyplot as plt
import numpy as np
import math


class Neuron:
    def __init__(self, i, j):
        self.i = i
        self.j = j
        self.weights = None  # равномерное распределение

    def set_weights(self, flag, N=0, w=None):
        if (flag == True):
            self.weights = self._normalization(np.random.uniform(-0.5, 0.5, N))
        else:
            self.weights = w

    def _normalization(self, mass):
        length = ((mass**2).sum())**0.5
        mass = mass/length
        return mass


class SOM:
    """Self-Organizing Map — карта Кохонена"""

    def __init__(self, h_map, w_map, input_N, T, sigma=2.24):
        self.h_map = h_map
        self.w_map = w_map
        self.input_N = input_N
        self.map_neurons = self._create_neurons_map(h_map, w_map, input_N)
        self.win_neuron = None
        self.sigma = sigma
        self.T = T

    def _create_neurons_map(self, h_map, w_map, input_N):
        """Создание карты Кохонена"""
        map_n = [[0 for x in range(w_map)] for y in range(h_map)]
        for i in range(h_map):
            for j in range(w_map):
                map_n[i][j] = Neuron(i, j)
                map_n[i][j].set_weights(N=input_N, flag=True)
        return map_n

    def _get_learning_rate(self, t):
        """Скорость обучения"""
        return 0.5**(t/self.T)

    def _get_distanse(self, W, X):
        """Расстояние"""
        return (sum((W-X)**2))**0.5

    def _winner_neuron(self, X):
        """Запоминаем нейрон победитель"""
        win_neuron = self.map_neurons[0][0]
        d_min = self._get_distanse(win_neuron.weights, X)
        i_min, j_min = 0, 0

        for i in range(self.h_map):
            for j in range(1, self.w_map):
                d = self._get_distanse(self.map_neurons[i][j].weights, X)
                if(d < d_min):
                    d_min = d
                    i_min = i
                    j_min = j
                    win_neuron = self.map_neurons[i][j]
        self.win_neuron = Neuron(i_min, j_min)
        self.win_neuron.set_weights(flag=False, w=win_neuron.weights.copy())
        #print("i_min, j_min",i_min, j_min)
        return i_min, j_min

    def _tuning_weights_of_neurons(self, X, t):
        """Настройка весов"""
        for i in range(self.h_map):
            for j in range(self.w_map):
                W = self.map_neurons[i][j].weights.copy()
                g = self._get_learning_rate(
                    t) * self._neighborhood_function(t, i, j)
                self.map_neurons[i][j].weights = W + g*(X-W)

    def _get_sigma(self, t):
        """Сигма"""
        return self.sigma / (1 + t/self.T)

    def _neighborhood_function(self, t, i, j):
        """Функция соседства (сила связи между нейронами)"""
        arr = np.array([i, j])
        arr2 = np.array([self.win_neuron.i, self.win_neuron.j])
        d = self._get_distanse(arr, arr2)

        b = 1/(2*self._get_sigma(t)**2)
        return np.exp((-1)*b*(d**2))

    def _normalization(self, mass):
        """Нормализация вектора"""
        length = ((mass**2).sum())**0.5
        mass = mass/length
        return mass

    def fit(self, train_data):
        """Обучение"""
        t = 0
        train_data = self._normalization(train_data)

        for ep in range(self.T):
            train_copy = train_data.copy()
            while train_copy.shape[0]:
                random_index = int(np.random.choice(
                    np.arange(train_copy.shape[0])))
                self._winner_neuron(train_copy[random_index])
                self._tuning_weights_of_neurons(train_copy[random_index], t)

                train_copy = np.delete(train_copy, (random_index), axis=0)
            t += 1

    def predict(self, test_data):
        """Предсказание"""
        map_result = [["" for x in range(self.w_map)]
                      for y in range(self.h_map)]
        for ind, el in enumerate(test_data):
            i, j = self._winner_neuron(el)
            # print(i,j)
            map_result[i][j] += ""+str(ind)+" "
        return map_result

# Нормализация входных данных
# Нормализация весов


model = SOM(input_N=29, h_map=10, w_map=10, T=30)

# Размер маленький, средний, крупный.
# Имеет 2 ноги, 4 ноги, шерсть, копыта, гриву, перья.
# Любит охоту, бегать, летать, плавать.

train_data = [
    [1, 0, 0,  1, 0, 0, 0, 0, 1,  0, 0, 1, 0],
    [1, 0, 0,  1, 0, 0, 0, 0, 1,  0, 0, 0, 0],
    [1, 0, 0,  1, 0, 0, 0, 0, 1,  0, 0, 0, 1],
    [1, 0, 0,  1, 0, 0, 0, 0, 1,  0, 0, 1, 1],
    [1, 0, 0,  1, 0, 0, 0, 0, 1,  1, 0, 1, 0],
    [1, 0, 0,  1, 0, 0, 0, 0, 1,  1, 0, 1, 0],
    [0, 1, 0,  1, 0, 0, 0, 0, 1,  1, 0, 1, 0],
    [0, 1, 0,  0, 1, 1, 0, 0, 0,  1, 0, 0, 0],
    [0, 1, 0,  0, 1, 1, 0, 0, 0,  0, 1, 0, 0],
    [0, 1, 0,  0, 1, 1, 0, 1, 0,  1, 1, 0, 0],
    [1, 0, 0,  0, 1, 1, 0, 0, 0,  1, 0, 0, 0],
    [0, 0, 1,  0, 1, 1, 0, 0, 0,  1, 1, 0, 0],
    [0, 0, 1,  0, 1, 1, 0, 1, 0,  1, 1, 0, 0],
    [0, 0, 1,  0, 1, 1, 1, 1, 0,  0, 1, 0, 0],
    [0, 0, 1,  0, 1, 1, 1, 1, 0,  0, 1, 0, 0],
    [0, 0, 1,  0, 1, 1, 1, 0, 0,  0, 0, 0, 0]
]
y = [
    "Голубь", "Курица", "Утка", "Гусь", "Сова", "Ястреб", "Орел", "Лиса",
    "Собака", "Волк", "Кошка", "Тигр", "Лев", "Лошадь", "Зебра", "Корова"]


for i in range(len(train_data)):
    train_data[i] = [0.2 if j == i else 0 for j in range(
        len(train_data))]+train_data[i]
train_data = np.array(train_data)


model.fit(train_data)
map_res = model.predict(train_data)


for i in range(len(map_res)):
    for j in range(len(map_res[0])):
        if(map_res[i][j] != ''):
            l = map_res[i][j].rstrip().split(" ")
            map_res[i][j] = ""
            for u in range(len(l)):
                int_s = int(l[u])
                map_res[i][j] += y[int_s]
        else:
            map_res[i][j] = "·"
np.set_printoptions(linewidth=150)
map_res = np.array(map_res)
max_len = max([len(str(e)) for r in map_res for e in r])
for row in map_res:
    print(*list(map('{{:>{length}}}'.format(length=max_len).format, row)))
