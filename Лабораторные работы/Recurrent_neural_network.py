#!/usr/bin/env python
# coding: utf-8

import numpy as np


class HopfieldNetwork:
    def __init__(self, w_input, h_input):
        self.weights = None
        self.remembered_images = []
        self.neurons_count = w_input*h_input
        self.w_input = w_input

    def _calculate_pseudoinverse_matrix(self, F):
        '''Вычисление псевдоинверсной матрицы'''
        F_t = F.transpose()
        M_inv = np.linalg.inv((np.dot(F_t, F)))
        F_plus = np.dot(M_inv, F_t)
        return F_plus

    def _reshape(self, data):
        data = np.array(data).transpose()
        return data

    def signum_func(self, net):
        if net > 0:
            return 1
        elif net < 0:
            return -1
        else:
            return 0  # означает, что нужно сохранить предыдущее значение выхода

    def fit(self, train_data):
        train_data = self._reshape(train_data)
        train_data_plus = self._calculate_pseudoinverse_matrix(train_data)
        self.weights = np.dot(train_data, train_data_plus)
        np.fill_diagonal(self.weights, 0)

    def _pred(self, td):
        t = 1
        signum_func_vectorize = np.vectorize(self.signum_func)
        td = np.array(td).transpose()
        self._print_(t-1, td)
        while(True):
            td_temp = signum_func_vectorize(np.dot(self.weights, td))
            if(0 in td_temp):
                j, z = np.where(td_temp == 0)
                if (j.size and z.size) != 0:
                    for m in range(j.size):
                        td_temp[j][z] = td[j][z]
            if(np.array_equal(td, td_temp)):
                return td
            else:
                td = td_temp
            self._print_(t, td_temp)
            t += 1
        self._print_(t, td)

#     def _pred(self,td):
#         t=1
#         td = np.array(td).transpose()
#         self._print_(t-1,td)
#         td_temp = td.copy()
#         while(True):
#             for i in range(self.neurons_count):
#                 y_new = np.dot(self.weights[i],td_temp)
#                 y_new = self.signum_func(y_new)
#                 print("test",y_new, td_temp[i])
#                 if(y_new!=0):
#                     td_temp[i] = y_new
#                 if(np.array_equal(td,td_temp)):
#                     print("РАвны")
#                     return
#                 td  = td_temp

    def prediction(self, test_data):
        print(test_data)
        for el in test_data:
            self._pred(el)

    def _print_(self, t, data):
        print(str(t)+"  ")
        for i in range(0, data.size, self.w_input):
            l = data[i:i+self.w_input]
            l = [" " if j == -1 else "@" for j in l]
            print(l)
        print("\n")


train_obr = []
obraz0 = [-1, -1, -1, -1, -1, -1, -1,   -1, 1, 1, 1, 1, 1, -1,   -1, 1, -1, -1, -1, 1, -1,   -1, 1, -1, -1, -1, 1, -1, -1, 1, -
          1, -1, -1, 1, -1, -1, 1, -1, -1, -1, 1, -1, -1, 1, -1, -1, -1, 1, -1,   -1, 1, 1, 1, 1, 1, -1,   -1, -1, -1, -1, -1, -1, -1]
train_obr.append(obraz0)
obraz1 = [-1, -1, -1, -1, -1, -1, -1,   -1, -1, -1, 1, -1, -1, -1,   -1, -1, 1, 1, -1, -1, -1,   -1, -1, -1, 1, -1, -1, -1,   -1, -
          1, -1, 1, -1, -1, -1,   -1, -1, -1, 1, -1, -1, -1,   -1, -1, -1, 1, -1, -1, -1,    -1, 1, 1, 1, 1, 1, -1,   -1, -1, -1, -1, -1, -1, -1]
train_obr.append(obraz1)
obraz2 = [-1, -1, -1, -1, -1, -1, -1,   -1, 1, 1, 1, 1, 1, -1,   -1, -1, -1, -1, -1, 1, -1,    -1, -1, -1, -1, -1, 1, -1,   -1, 1,
          1, 1, 1, 1, -1,   -1, 1, -1, -1, -1, -1, -1,    1, 1, -1, -1, -1, -1, -1,   -1, 1, 1, 1, 1, 1, -1,   -1, -1, -1, -1, -1, -1, -1]
train_obr.append(obraz2)
obraz3 = [-1, -1, -1, -1, -1, -1, -1,   -1, 1, 1, 1, 1, 1, -1,   -1, -1, -1, -1, -1, 1, -1,     -1, -1, -1, -1, -1, 1, -1,   -1, 1,
          1, 1, 1, 1, -1,   -1, -1, -1, -1, -1, 1, -1,   -1, -1, -1, -1, -1, 1, -1,    -1, 1, 1, 1, 1, 1, -1,   -1, -1, -1, -1, -1, -1, -1]
train_obr.append(obraz3)
obraz4 = [-1, -1, -1, -1, -1, -1, -1,   -1, 1, -1, -1, -1, 1, -1,   -1, 1, -1, -1, -1, 1, -1,    -1, 1, -1, -1, -1, 1, -1,   -1, 1,
          1, 1, 1, 1, -1,   -1, -1, -1, -1, -1, 1, -1,    -1, -1, -1, -1, -1, 1, -1,  -1, -1, -1, -1, -1, 1, -1,   -1, -1, -1, -1, -1, -1, -1]
train_obr.append(obraz4)
obraz5 = [-1, -1, -1, -1, -1, -1, -1,   -1, 1, 1, 1, 1, 1, -1,   -1, 1, -1, -1, -1, -1, -1,   -1, 1, -1, -1, -1, -1, -1,    -1, 1,
          1, 1, 1, 1, -1,  -1, -1, -1, -1, -1, 1, -1,   -1, -1, -1, -1, -1, 1, -1,   -1, 1, 1, 1, 1, 1, -1,  -1, -1, -1, -1, -1, -1, -1]
train_obr.append(obraz5)
obraz6 = [-1, -1, -1, -1, -1, -1, -1,   -1, 1, 1, 1, 1, 1, -1,   -1, 1, -1, -1, -1, -1, -1,   -1, 1, -1, -1, -1, -1, -1,    -1, 1,
          1, 1, 1, 1, -1,   -1, 1, -1, -1, -1, 1, -1,   -1, 1, -1, -1, -1, 1, -1,   -1, 1, 1, 1, 1, 1, -1,   -1, -1, -1, -1, -1, -1, -1]
train_obr.append(obraz6)


test_obr = []
test_obraz1 = [1, -1, -1, -1, -1, -1, -1,     1, 1, 1, 1, 1, -1, -1,     -1, -1, -1, 1, -1, -1, -1,   -1, -1, -1, 1, 1, -1, -1, -
               1, -1, 1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1,    -1, 1, 1, 1, 1, 1, -1,   -1, -1, -1, -1, -1, -1, 1]
test_obr.append(test_obraz1)
test_obraz0 = [1, 1, -1, -1, -1, 1, -1,   -1, 1, 1, 1, 1, 1, -1,   -1, 1, 1, -1, 1, 1, -1,   -1, 1, -1, -1, -1, 1, -1, -1,
               1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1,   -1, 1, 1, 1, 1, 1, -1,   -1, -1, -1, 1, -1, -1, -1]
test_obr.append(test_obraz0)
test_obraz4 = [-1, 1, -1, -1, -1, -1, -1,   -1, 1, -1, -1, -1, 1, -1,   -1, 1, -1, -1, -1, 1, -1,    -1, 1, 1, -1, 1, 1, -1,   -1, 1,
               1, 1, 1, 1, -1,   -1, -1, -1, 1, -1, 1, -1,    -1, -1, 1, -1, -1, 1, -1,  -1, -1, -1, -1, -1, 1, -1,   -1, -1, -1, -1, -1, -1, -1]
test_obr.append(test_obraz4)

b = 7
for i in range(0, len(test_obraz4), 7):
    l = test_obraz4[i:i+b]
    l = [" " if j == -1 else "@" for j in l]
    print(l)


model = HopfieldNetwork(w_input=7, h_input=9)
model.fit(train_obr)


# In[116]:

b = 7
for i in range(0, t.size, 7):
    l = t[i:i+b]
    l = [" " if j == -1 else "@" for j in l]
    print(l)


a = np.array([[1], [0], [1]])
b = np.array([[1], [0], [1]])
np.array_equal(a, b)


W = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

a = np.array([[1], [0], [1]])
b = np.dot(W, a)
print(b)


b = np.array([[4], [10], [16], [4], [4]])

if(4 in b):
    print(True)
    i, j = np.where(b == 100)
    print(i, j)
else:
    print(False)


W = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
np.fill_diagonal(W, 0)
print(W)


def signum_func(net):
    if net > 0:
        return 1
    elif net < 0:
        return -1
    else:
        return 0  # означает, что нужно сохранить предыдущее значение выхода


signum_func_vectorize = np.vectorize(signum_func)


b = np.array([[-1], [-1], [1], [1], [1]])

s = signum_func_vectorize(b)
print(s)
