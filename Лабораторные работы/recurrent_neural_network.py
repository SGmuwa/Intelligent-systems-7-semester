# Рекуррентная сеть Эльмана

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


# ⬜⬛
answers = [
    "⬜⬜⬜⬜⬜⬜⬜\n" +
    "⬜⬛⬛⬛⬛⬛⬜\n" +
    "⬜⬛⬜⬜⬜⬛⬜\n" +
    "⬜⬛⬜⬜⬜⬛⬜\n" +
    "⬜⬛⬜⬜⬜⬛⬜\n" +
    "⬜⬛⬜⬜⬜⬛⬜\n" +
    "⬜⬛⬜⬜⬜⬛⬜\n" +
    "⬜⬛⬛⬛⬛⬛⬜\n" +
    "⬜⬜⬜⬜⬜⬜⬜",

    "⬜⬜⬜⬜⬜⬜⬜\n" +
    "⬜⬜⬜⬛⬜⬜⬜\n" +
    "⬜⬜⬛⬛⬜⬜⬜\n" +
    "⬜⬜⬜⬛⬜⬜⬜\n" +
    "⬜⬜⬜⬛⬜⬜⬜\n" +
    "⬜⬜⬜⬛⬜⬜⬜\n" +
    "⬜⬜⬜⬛⬜⬜⬜\n" +
    "⬜⬛⬛⬛⬛⬛⬜\n" +
    "⬜⬜⬜⬜⬜⬜⬜",

    "⬜⬜⬜⬜⬜⬜⬜\n" +
    "⬜⬛⬛⬛⬛⬛⬜\n" +
    "⬜⬜⬜⬜⬜⬛⬜\n" +
    "⬜⬜⬜⬜⬜⬛⬜\n" +
    "⬜⬛⬛⬛⬛⬛⬜\n" +
    "⬜⬛⬜⬜⬜⬜⬜\n" +
    "⬜⬛⬜⬜⬜⬜⬜\n" +
    "⬜⬛⬛⬛⬛⬛⬜\n" +
    "⬜⬜⬜⬜⬜⬜⬜",

    "⬜⬜⬜⬜⬜⬜⬜\n" +
    "⬜⬛⬜⬜⬜⬛⬜\n" +
    "⬜⬛⬜⬜⬜⬛⬜\n" +
    "⬜⬛⬜⬜⬜⬛⬜\n" +
    "⬜⬛⬛⬛⬛⬛⬜\n" +
    "⬜⬜⬜⬜⬜⬛⬜\n" +
    "⬜⬜⬜⬜⬜⬛⬜\n" +
    "⬜⬜⬜⬜⬜⬛⬜\n" +
    "⬜⬜⬜⬜⬜⬜⬜",

    "⬜⬜⬜⬜⬜⬜⬜\n" +
    "⬜⬛⬛⬛⬛⬛⬜\n" +
    "⬜⬛⬜⬜⬜⬜⬜\n" +
    "⬜⬛⬜⬜⬜⬜⬜\n" +
    "⬜⬛⬛⬛⬛⬛⬜\n" +
    "⬜⬜⬜⬜⬜⬛⬜\n" +
    "⬜⬜⬜⬜⬜⬛⬜\n" +
    "⬜⬛⬛⬛⬛⬛⬜\n" +
    "⬜⬜⬜⬜⬜⬜⬜",

    "⬜⬜⬜⬜⬜⬜⬜\n" +
    "⬜⬛⬛⬛⬛⬛⬜\n" +
    "⬜⬛⬜⬜⬜⬜⬜\n" +
    "⬜⬛⬜⬜⬜⬜⬜\n" +
    "⬜⬛⬛⬛⬛⬛⬜\n" +
    "⬜⬛⬜⬜⬜⬛⬜\n" +
    "⬜⬛⬜⬜⬜⬛⬜\n" +
    "⬜⬛⬛⬛⬛⬛⬜\n" +
    "⬜⬜⬜⬜⬜⬜⬜"
]

tests = [
    #1:
    "⬛⬜⬜⬜⬜⬜⬜\n" +
    "⬛⬛⬛⬛⬛⬜⬜\n" +
    "⬜⬜⬜⬛⬜⬜⬜\n" +
    "⬜⬜⬜⬛⬛⬜⬜\n" +
    "⬜⬜⬛⬛⬜⬜⬜\n" +
    "⬜⬜⬜⬛⬜⬜⬜\n" +
    "⬜⬜⬜⬛⬛⬜⬜\n" +
    "⬜⬛⬛⬛⬛⬛⬜\n" +
    "⬜⬜⬜⬜⬜⬜⬛",
    #0:
    "⬛⬛⬜⬜⬜⬛⬜\n" +
    "⬜⬛⬛⬛⬛⬛⬜\n" +
    "⬜⬛⬛⬜⬛⬛⬜\n" +
    "⬜⬛⬜⬜⬜⬛⬜\n" +
    "⬜⬛⬜⬜⬜⬛⬜\n" +
    "⬜⬛⬛⬜⬛⬛⬜\n" +
    "⬜⬛⬜⬜⬜⬛⬛\n" +
    "⬜⬛⬛⬛⬛⬛⬜\n" +
    "⬜⬜⬜⬛⬜⬜⬜",
    #4:
    "⬜⬛⬜⬜⬜⬜⬜\n" +
    "⬜⬛⬜⬜⬜⬛⬜\n" +
    "⬜⬛⬜⬜⬜⬛⬜\n" +
    "⬜⬛⬛⬜⬛⬛⬜\n" +
    "⬜⬛⬛⬛⬛⬛⬜\n" +
    "⬜⬜⬜⬛⬜⬛⬜\n" +
    "⬜⬜⬛⬜⬜⬛⬜\n" +
    "⬜⬜⬜⬜⬜⬛⬜\n" +
    "⬜⬜⬜⬜⬜⬜⬜"
]

def convertStringToIntArray(string):
    return [(1 if s == '⬛' else -1) for s in string]

# Конвектирует string в Array<int>.


def convertInputFromStringToIntArrays(inputDataDictionary):
    outputData = [None] * len(inputDataDictionary)
    for key, string in enumerate(inputDataDictionary):
        outputData[key] = convertStringToIntArray(string.replace('\n', ''))
    return outputData

answers_num = convertInputFromStringToIntArrays(answers)
tests_num = convertInputFromStringToIntArrays(tests)

model = HopfieldNetwork(w_input=7, h_input=9)
model.fit(answers_num)
print(tests[model.prediction(tests_num[0])])
