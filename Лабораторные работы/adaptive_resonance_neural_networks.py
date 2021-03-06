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
# «Адаптивные резонансные нейронные сети»
# 1. Продолжение рекуррентных сетей.
# 2. Решает проблему стирания старых образов при запоминании новых.
# 3. Мы используем АРТ1 (бинарный вход), хотя есть
#    АРТ2 (входной вектор из непрерывных величин)
# 4. если входной вектор похож на один из ранее запомненных векторов, то
#    запомненный вектор будет изменяться, чтобы стать похожим на входной вектор.
# 5. возможен отказ от классификации, если входной вектор не похож ни на
#    один из запомненных векторов. В этой ситуации создаётся новый класс.

from functools import reduce
import random
BETA = 1.0  # Бета-параметр. Размер кластера
VIGILANCE = 0.9  # 0 <= VIGILANCE < 1. Параметр внимательности
max_items = 11  # Количество товаров
max_customers = 10  # Количество покупателей
total_prototype_vectors = 5  # Количество кластеров
prototype_vectors = []  # Векторы прототипов для каждого кластера
members = []  # Количество членов в кластерах
# Номер кластера, к которому принадлежит покупатель
membersships = [-1]*max_customers
sum_vector = []  # Вектор суммирования для выдачи рекомендаций
itemName = ["Молоток", "Бумага", "Кроссовки", "Отвертка", "Ручка", "Кит-Кат",
            "Гаечный ключ", "Карандаш", "Heath Bar", "Рулетка", "Связующее вещество"]
data = [[0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],  # Массив векторов признаков. Поля представляют товар, который приобретет покупатель.
        [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]]

MAX_ITEMS = 11
MAX_CUSTOMERS = 10
TOTAL_PROTOTYPE_VECTORS = 5

BETA = 1.0  # Бета-параметр. Размер кластера
VIGILANCE = 0.9  # 0 <= VIGILANCE < 1. Параметр внимательности

DATABASE = []
PROTOTYPES = []

VERBOSITY = False

print('p# — прототип. c# — покупатель.')


def bit_wise_and(v1, v2):
    return Vector([item[0] & item[1] for item in zip(v1, v2)])


class Vector(list):

    def __init__(self, init=None, rand=False):
        if not init:
            if rand:
                init = [random.randint(0, 1) for _ in range(MAX_ITEMS)]
            else:
                init = [0] * MAX_ITEMS

        super().__init__(init)

    def __eq__(self, other):
        return self.id == other.id

    @property
    def magn(self):
        return float(sum(self))


class Prototype(Vector):
    _id = 0

    def __init__(self, customer):
        Prototype._id += 1
        self.id = self._id
        super().__init__(list(customer))
        self.customers = []
        self.changed = False
        self.add_customer(customer)

    def __repr__(self):
        l = super(Prototype, self).__repr__()
        return 'p#%2i %s' % (self.id, l)

    def add_customer(self, customer):
        if customer.cluster and customer.cluster == self:
            return

        if customer.cluster:
            customer.cluster.remove_customer(customer)

        self.customers.append(customer)
        customer.cluster = self
        self.update()

    def remove_customer(self, customer):
        self.customers.remove(customer)
        customer.cluster = None

        if not self.customers:
            PROTOTYPES.remove(self)
        else:
            self.update()

    def update(self):
        v = zip(*self.customers)

        for i, row in enumerate(v):
            self[i] = reduce(lambda a, b: a & b, row)

    @property
    def sum_vector(self):
        v = zip(*self.customers)
        return [sum(item) for item in v]


class Customer(Vector):
    _id = 0

    def __init__(self, *args, **kwargs):
        Customer._id += 1
        self.id = self._id
        super().__init__(*args, **kwargs)
        self.cluster = None

    def __repr__(self):
        l = super(Customer, self).__repr__()
        return 'c#%2i %s' % (self.id, l)

    def recomedation(self):
        if not self.cluster:
            return "None"

        max_val = -1
        recomedation = []

        for i, item in enumerate(self.cluster.sum_vector):
            if not self[i]:
                if item > max_val:
                    max_val = item
                    recomedation = [i]
                elif item == max_val:
                    recomedation.append(i)
        return recomedation


def init():
    for i in range(MAX_CUSTOMERS):
        DATABASE.append(Customer(data[i]))


def performART1():
    done = False
    count = 50
    while not done:
        done = True

        for customer in DATABASE:
            # Пытаемся найти подходящий кластер
            for prototype in PROTOTYPES:
                if customer.cluster and customer.cluster == prototype:
                    continue

                and_result = bit_wise_and(customer, prototype)
                result = and_result.magn / (BETA + prototype.magn)
                test = customer.magn / (BETA + MAX_ITEMS)

                # Проверка на схожесть
                if result > test:
                    # Тест на внимательность
                    if and_result.magn / customer.magn < VIGILANCE:
                        done = False
                        prototype.add_customer(customer)

            # Создаем новый кластер для неопределенного вектора
            if not customer.cluster:
                done = False
                if len(PROTOTYPES) < TOTAL_PROTOTYPE_VECTORS:
                    new_prototype = Prototype(customer)
                    PROTOTYPES.append(new_prototype)

        count -= 1
        if count <= 0:
            break


if __name__ == "__main__":
    init()
    performART1()
    for prototype in PROTOTYPES:
        print(prototype)
        for c in prototype.customers:
            print(c, c.recomedation())
        print('-------------------------------')
        print()
