#!/usr/bin/python3
# coding: utf-8
# Сеть радиально-базисных функций Radial Basis Function Network RBFN or RBF
# Главное отличие RBF-сетей от обычных многослойных сетей прямого распространения состоит в функции нейронов скрытого слоя. В обычной многослойной сети каждый нейрон рабочего слоя реализует в многомерном пространстве гиперплоскость (рис. 2а), а RBF-нейрон – гиперсферу (рис. 2б, 2в).
# 1. Пытаются разбить пространство наблюдений гиперсферами. Гиперсфера задаётся центром и радиусом. Каждый нейрон скрытого слоя реализует в многомерном пространстве гиперсферу.
# 2. Каждый объект классифицируется к классу тем, где его ближайший центр.
# 3. Есть входной слой, есть скрытый слой, есть выходной слой.
# Скрытый — это радиально-базисный слой. В большинство реализаций скрытый слой имеет больший размер, чем входной слой. Каждый скрытый нейрон символизирует центр некоторой гипер-сферы кластера данных. Выход нейрона — это реакция на расстояние от наблюдения до центра. Вместо весов координаты центра. Выходной слой это один или несколько нейронов.
# 4. Активационная функция скрытых нейронов exp(-abs(x-x[i])²/(2σ[i]²)). Если расстояние от наблюдения до центра велико, то функция будет уменьшаться. 
# 5. Обучение сети РБФ. Сначала определяются центры и отклонения (радиус) для радиальных нейронов. Настройка параметров линейного выходного слоя. Расположение центров должно соответсвовать кластерам реально присутствующих в исходных данных.
# 6. Использование: хорошо работает только с наблюдениями, лежащими близко к обучающим наблюдениям. При удалении от обучающего множества значение функции отклика быстро спадает до нуля. Используется в задачах регрессии и классификации.
# Функция отклика — Выход скрытого слоя?

# abs(x - c[i]) = √((x[1] - c[1])² + (x[2] - c[2])² + ... + (x[n] - c[n])²)
# c[i] — нейрон. x — вход.

# https://basegroup.ru/community/articles/rbf



import numpy as np
import matplotlib.pyplot as plt


class RadialBasisFunctionNetworks:
    '''Класс реализует сеть радиальнобазисных функций'''

    def __init__(self, input_shape, hidden_shape, output_shape, n=0.2, T=20, R=4):
        self.input_shape = input_shape  # Количество входных нейронов
        self.hidden_shape = hidden_shape  # Количество скрытых нейронов
        self.output_shape = output_shape  # Количество выходных нейронов

        self.weights = None
        self.centers = None
        self.sigma_mass = None  # Ширина окна активационной функции

        self.n = n  # Скорость обучения
        self.T = T  # Коэффициент уменьшения n
        self.R = R  # Количество близжайших соседей

    def _radial_basis_function(self, i_c, center, data_vector):
        '''Гауссова функция'''
        # np.linalg.norm(data_vector-center)**2
        sum = 0
        for i in range(self.input_shape):
            sum += (data_vector[i]-center[i])**2
        return np.exp(sum/(2 * (self.sigma_mass[i_c])**2))

    def _calculate_interpolation_matrix(self, X):
        '''Вычисление интерполяционной матрицы'''
        F = np.zeros((X.shape[0], self.hidden_shape))
        for ind_X, x in enumerate(X):
            for ind_c, center in enumerate(self.centers):
                F[ind_X, ind_c] = self._radial_basis_function(ind_c, center, x)
        return F

    def _calculate_pseudoinverse_matrix(self, F):
        '''Вычисление псевдоинверсной матрицы'''
        F_t = F.transpose()
        M_inv = np.linalg.inv((np.dot(F_t, F)))
        F_plus = np.dot(M_inv, F_t)
        return F_plus

    def _k_means(self, x):
        '''Алгоритм k-means для нахождения исходных центров данных'''

        # Уточнение центров скрытых нейронов
        random_index = np.random.permutation(x.shape[0]).tolist()
        self.centers = np.array([x[ind]
                                 for ind in random_index][:self.hidden_shape])
        for i, el in enumerate(x):
            d = np.array([])
            for j_c, c in enumerate(self.centers):
                d = np.append(d, (np.sum((el-c)**2))**(1/2))
            min = np.argmin(d)
            self.centers[min] = self.centers[min] + \
                self.n*(el-self.centers[min])
            self.n = self.n/(1+(i/self.T))

        self.sigma_mass = np.array([])
        # Расчет сигма для скрытых нейронов (массив)
        for i in range(self.centers.shape[0]):
            c = self.centers[i]
            d = np.empty((0, 2), float)
            for j in range(self.centers.shape[0]):
                if(j != i):
                    s = (np.sum((c-self.centers[j])**2))**(1/2)
                    d = np.append(d, [[j, s]], axis=0)
            d.view('i8,i8').sort(order=['f1'], axis=0)
            # R близжайших соседей
            d = d[:self.R]

            su = 0
            for r in range(d.shape[0]):
                su += (np.sum(self.centers[r]-c)**2)
            su = (su/self.R)**(1/2)
            self.sigma_mass = np.append(self.sigma_mass, [su])
        print("Σ mass:", self.sigma_mass)

    def fit(self, X, Y):
        '''Обучение'''
        # Случайно выбираем центры из начальных данных
#         random_index = np.random.permutation(X.shape[0]).tolist()
#         self.centers = [X[ind] for ind in random_index][:self.hidden_shape]
        self._k_means(X)
        F = self._calculate_interpolation_matrix(X)
        F_plus = self._calculate_pseudoinverse_matrix(F)
        self.weights = np.dot(F_plus, Y)

    def predict(self, X):
        '''Вычисляет по обучающей выборке вектор ответов Y'''
        F = self._calculate_interpolation_matrix(X)
        predictions = np.dot(F, self.weights)
        return predictions


# Чтение данных из файла
x = []
# Ирисы описываются по 4 критериям. Ширины, длины...
# Ирисы Фишера — это набор данных для задачи классификации, на примере которого Рональд Фишер в 1936 году продемонстрировал работу разработанного им метода дискриминантного анализа.
# https://ru.wikipedia.org/wiki/Ирисы_Фишера
f = open("Irisy.txt", encoding='utf-8')
for line in f:
    l = line.split(' ')
    for i in range(len(l)):
        l[i] = float(l[i].strip(" ").lower().strip("\n"))
    x.append(l)
x = np.array(x)


y = []
# Ответы ирисов
f = open("y.txt", encoding='utf-8')
for line in f:
    l = line.split(' ')
    for i in range(len(l)):
        l[i] = float(l[i].strip(" ").lower().strip("\n"))
    y.append(l)
y = np.array(y)


# Считать x и y
model = RadialBasisFunctionNetworks(
    input_shape=4, hidden_shape=10, output_shape=3)
model.fit(x, y)
y_pred = model.predict(x)

# Подсчёт точности
count = 0
for i, el in enumerate(y_pred):
    if(np.argmax(el) == np.argmax(y[i])):
        count += 1


print("Точность", round((count/x.shape[0])*100), "%")


# # Тестирование сети для данных о кластеризации по вероятности страхового случая


# Чтение данных из файла
x = []
# данные зависимости потребления Y (усл. ед.) от дохода X (усл.ед.) для некоторых домашних хозяйств.
f = open("data_for_k_means.txt", encoding='utf-8')
for line in f:
    l = line.split(' ')
    for i in range(len(l)):
        l[i] = float(l[i].strip(" ").lower().strip("\n"))
    x.append(l)
x = np.array(x)


y = []
f = open("y_k_means.txt", encoding='utf-8')
for line in f:
    l = line.split(' ')
    for i in range(len(l)):
        l[i] = float(l[i].strip(" ").lower().strip("\n"))
    y.append(l)
y = np.array(y)


# Считать x и y
model2 = RadialBasisFunctionNetworks(
    input_shape=4, hidden_shape=10, output_shape=4)
model2.fit(x, y)
y_pred = model2.predict(x)


count = 0
for i, el in enumerate(y_pred):
    if(np.argmax(el) == np.argmax(y[i])):
        count += 1


print("Точность", round((count/x.shape[0])*100), "%")


# # Тестирование сети на данных из лабы Backpropagation


train_data = [
    ([2, 0, 0, 0], [0, 0, 1, 0]),
    ([2, 0, 0, 1], [0, 0, 1, 0]),
    ([2, 0, 1, 1], [1, 0, 0, 0]),
    ([2, 0, 1, 2], [1, 0, 0, 0]),
    ([2, 1, 0, 2], [0, 0, 0, 1]),
    ([2, 1, 0, 1], [1, 0, 0, 0]),
    ([1, 0, 0, 0], [0, 0, 1, 0]),
    ([1, 0, 0, 1], [0, 0, 0, 1]),
    ([1, 0, 1, 1], [1, 0, 0, 0]),
    ([1, 0, 1, 2], [0, 0, 0, 1]),
    ([1, 1, 0, 2], [0, 0, 0, 1]),
    ([1, 1, 0, 1], [0, 0, 0, 1]),

    ([0, 0, 0, 0], [0, 0, 1, 0]),
    ([0, 0, 0, 1], [0, 0, 0, 1]),
    ([0, 0, 1, 1], [0, 0, 0, 1]),
    ([0, 0, 1, 2], [0, 1, 0, 0]),
    ([0, 1, 0, 2], [0, 1, 0, 0]),
    ([0, 1, 0, 1], [0, 0, 0, 1]),
]

test_data = [
    ([2, 1, 1, 1], [1, 0, 0, 0]),
    ([1, 1, 1, 2], [0, 0, 0, 1]),
    ([0, 0, 0, 0], [0, 0, 1, 0]),
    ([0, 1, 1, 1], [0, 0, 0, 1]),
    ([2, 0, 1, 3], [0, 0, 0, 1]),
    ([2, 1, 0, 3], [0, 0, 0, 1]),
    ([0, 1, 0, 3], [0, 1, 0, 0]),

]

x, y = np.array([l for l, y in train_data]), np.array(
    [y for l, y in train_data])
x_test, y_test = np.array([l for l, y in test_data]
                          ), np.array([y for l, y in test_data])


model3 = RadialBasisFunctionNetworks(
    input_shape=4, hidden_shape=15, output_shape=4)
model3.fit(x, y)
y_pred = model3.predict(x)

count = 0
for i, el in enumerate(y_pred):
    if(np.argmax(el) == np.argmax(y[i])):
        count += 1
print("Точность", round((count/x.shape[0])*100), "%")


y_pred = model3.predict(x_test)

count = 0
for i, el in enumerate(y_pred):
    print(np.argmax(el), np.argmax(y_test[i]))  # по строкам
    if(np.argmax(el) == np.argmax(y_test[i])):
        count += 1
print("Точность", round((count/x_test.shape[0])*100), "%")


n = 0.2
R = 4  # Количество близжайших соседей
T = 20

# Уточнение центров скрытых нейронов
random_index = np.random.permutation(x.shape[0]).tolist()
centers = np.array([x[ind] for ind in random_index][:20])
for i, el in enumerate(x):
    d = np.array([])
    for j_c, c in enumerate(centers):
        d = np.append(d, (np.sum((el-c)**2))**(1/2))
    min = np.argmin(d)
    centers[min] = centers[min] + n*(el-centers[min])
    n = n/(1+(i/T))


sigma_mass = np.array([])
# Расчет сигма для скрытых нейронов (массив)
for i in range(centers.shape[0]):
    c = centers[i]
    d = np.empty((0, 2), float)
    for j in range(centers.shape[0]):
        if(j != i):
            s = (np.sum((c-centers[j])**2))**(1/2)
            d = np.append(d, [[j, s]], axis=0)

    d.view('i8,i8').sort(order=['f1'], axis=0)
    # R близжайших соседей
    d = d[:R]

    su = (sum(np.sum(centers[r]-c)**2 for r in range(d.shape[0])) / R) ** 0.5
    sigma_mass = np.append(sigma_mass, [su])
print("Σ mass:", sigma_mass)
