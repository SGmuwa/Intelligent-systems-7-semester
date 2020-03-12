# Метод обратного распространения ошибки

import numpy as np
import sys

INPUT_NEURON = 4
HID_NEURON = 3
OUTPUT_NEURON = 4


class Neurocontroller:
    def __init__(self, learning_rate=0.2):
        # равномерное распределение
        self.weights_0_1 = np.random.uniform(-0.5, 0.5, (5, 3))
        # равномерное распределение
        self.weights_1_2 = np.random.uniform(-0.5, 0.5, (4, 4))
        # Позволяет пробежаться по вектору, и к каждому элементу применить сигмоидную функцию
        self.sigmoid_mapper = np.vectorize(self.sigmoid)
        self.learning_rate = np.array([learning_rate])

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, val):
        return (val * (1 - val))

    # Прямой проход
    def predict(self, inputs):
        # Вычисляем вход в скрытый слой и выход из него
        inputs = np.append(inputs, 1)  # Дополнительный вход для смещения
        # Умножаем веса на входные значения
        inputs_1 = np.dot(inputs, self.weights_0_1)
        outputs_1 = self.sigmoid_mapper(inputs_1)
        # Вычисляем вход в выходной слой и выход из него
        outputs_1 = np.append(outputs_1, 1)  # Дополнительный вход для смещения
        inputs_2 = np.dot(outputs_1, self.weights_1_2)
        outputs_2 = self.sigmoid_mapper(inputs_2)
        return outputs_2

    def fit(self, inputs, target_predict):

        # Вычисляем вход в скрытый слой и выход из него
        inputs = np.append(inputs, 1)  # Дополнительный вход для смещения
        # Умножаем веса на входные значения
        inputs_1 = np.dot(inputs, self.weights_0_1)
        outputs_1 = self.sigmoid_mapper(inputs_1)

        # Вычисляем вход в выходной слой и выход из него
        outputs_1 = np.append(outputs_1, 1)  # Дополнительный вход для смещения
        inputs_2 = np.dot(outputs_1, self.weights_1_2)
        outputs_2 = self.sigmoid_mapper(inputs_2)
        fact_predict_out = outputs_2

        error_out = np.array([])
        # Вычислить ошибку для выходного слоя
        for o in range(OUTPUT_NEURON):
            error_out = np.append(
                error_out, (target_predict[o] - fact_predict_out[o]) * self.sigmoid_derivative(fact_predict_out[o]))

        sum = 0
        error_hid = np.array([])
        # Вычислить ошибку для срытого слоя
        for i in range(HID_NEURON):
            # Пройтись по строкам
            temp_sum = error_out * self.weights_1_2[i, :]
            sum = np.sum(temp_sum)
            error_hid = np.append(
                error_hid, self.sigmoid_derivative(outputs_1[i]) * sum)

        # Обновить веса для соединений скрытый слой - выходной слой
        # обновить смещения для нейронов выходного слоя
        # выходы из скрытого : outputs_1
        # oшибка выходного слоя : error_out
        for i in range(OUTPUT_NEURON):
            self.weights_1_2[:, i] += self.learning_rate*error_out[i]*outputs_1

        # Обновить веса для входной слой - скрытого слоя
        # Выходы из входного слоя inputs
        # Обновить смещения для нейронов скрытого слоя
        for i in range(HID_NEURON):
            self.weights_0_1[:, i] += self.learning_rate*error_hid[i]*inputs


def MSE(d, y):
    return 0.5 * np.sum((d-y)**2)


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

# атаковать бежать бродить/уворачиваться прятаться
action = ["Attack", "Run", "Wander", "Hide"]

# Обучение и тестирование нейроконтроллера
epochs = 500
learning_rate = 0.2
network = Neurocontroller(learning_rate=learning_rate)

for ep in range(epochs):
    inputs_ = []
    correct_predictions = []
    for input_stat, correct_predict in train_data:

        # Ошибка на входном векторе
        i_loss = MSE(network.predict(np.array(input_stat)),
                     np.array(correct_predict))
        #sys.stdout.write("\r {}, i_loss: {}".format(str(100 * ep/float(epochs))[:4], str(i_loss)[:5]))

        # Train
        network.fit(np.array(input_stat).T, np.array(correct_predict))

        inputs_.append(np.array(input_stat))
        correct_predictions.append(np.array(correct_predict))

    # print(correct_predictions)
    # Считаем ошибку после эпохи
    train_loss = 0
    for m in range(len(inputs_)):
        train_loss += MSE(network.predict(
            np.array(inputs_[m])), np.array(correct_predictions[m]))
    sys.stdout.write("\rProgress: {}, Training loss: {}".format(
        str(100 * ep/float(epochs))[:4], str(train_loss)[:5]))

# Проверяем сеть
for input_stat, correct_predict in train_data:
    print("For input: {} the prediction is: {}, expected: {}".format(
        str(input_stat),
        str(action[network.predict(np.array(input_stat)).argmax()]),
        str(action[np.array(correct_predict).argmax()])))

# Прогон на тестовых данных
# атаковать бежать бродить прятаться
action = ["Attack", "Run", "Wander", "Hide"]
test_data = [
    ([2, 1, 1, 1], [1, 0, 0, 0]),
    ([1, 1, 1, 2], [0, 0, 0, 1]),
    ([0, 0, 0, 0], [0, 0, 1, 0]),
    ([0, 1, 1, 1], [0, 0, 0, 1]),
    ([2, 0, 1, 3], [0, 0, 0, 1]),
    ([2, 1, 0, 3], [0, 0, 0, 1]),
    ([0, 1, 0, 3], [0, 1, 0, 0]),

]
for input_stat, correct_predict in test_data:
    print("For input: {} the prediction is: {}, expected: {}".format(
        str(input_stat),
        str(action[network.predict(np.array(input_stat)).argmax()]),
        str(action[np.array(correct_predict).argmax()])))
#         str(network.direct_distribution(np.array(input_stat))),
#         str(np.array(correct_predict))))

print('train_data', len(train_data))

weights_0_1 = np.random.uniform(-0.5, 0.5, (5, 3))
print(weights_0_1)
inputs = np.array([1, 2, 3, 4, 5])
inputs_1 = np.dot(inputs, weights_0_1)
print(inputs_1)

l = [0.10761236, 0.13300704, 0.08303307, 0.01834851, 0.26472631]
inputs = np.array([1, 2, 3, 4, 5])
l2 = l*inputs
print('l2.sum()', l2.sum())


def sigmoid_derivative(val):
    return (val * (1 - val))


target_predict = np.array([1, 1, 1, 1])
fact_predict_out = np.array([2, 3, 4, 5])
(target_predict - fact_predict_out) * sigmoid_derivative(fact_predict_out)

n = np.array([2, 0, 1, 1])
n = np.append(n, 1)
print(n)
weights_0_1 = np.random.uniform(-0.5, 0.5, (5, 3))
inputs_1 = np.dot(n, weights_0_1)
inputs_1


def MSE(d, y):
    return 0.5 * np.sum((d-y)**2)


d = np.array([0, 0, 1, 0])
y = np.array([2, 3, 4, 5])
#-2 -3 -3 -5
# 4  9  9  25
#
MSE(d, y)

weights_0_1 = np.random.uniform(-0.5, 0.5, (5, 3))
print(weights_0_1)
weights_0_1[:, 0] += 5*5*np.array([1, 1, 1, 1, 1])
print(weights_0_1)
