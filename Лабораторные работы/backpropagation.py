# Метод обратного распространения ошибки
# Мы получаем ошибку на последнем слое, ищем разницу, и отправляем разницу вглубь обратно.

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
    """Среднеквадратическая ошибка"""
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

action = ["Скупка     ", "Продажа    ", "Накапливать", "Тратить    "]

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

        # Train
        network.fit(np.array(input_stat).T, np.array(correct_predict))

        inputs_.append(np.array(input_stat))
        correct_predictions.append(np.array(correct_predict))

    # Считаем ошибку после эпохи
    train_loss = 0
    for m in range(len(inputs_)):
        train_loss += MSE(network.predict(
            np.array(inputs_[m])), np.array(correct_predictions[m]))
    sys.stdout.write("\r{}%, отклонение: {}".format(
        str(100 * ep/float(epochs))[:4], str(train_loss)[:5]))

# Проверяем сеть
print("\nДанные для обучения:")
for input_stat, correct_predict in train_data:
    print("Рынок: {} Инвестор: {} Ожидалось: {}".format(
        str(input_stat),
        str(action[network.predict(np.array(input_stat)).argmax()]),
        str(action[np.array(correct_predict).argmax()])))

# Прогон на тестовых данных
test_data = [
    ([2, 1, 1, 1], [1, 0, 0, 0]),
    ([1, 1, 1, 2], [0, 0, 0, 1]),
    ([0, 0, 0, 0], [0, 0, 1, 0]),
    ([0, 1, 1, 1], [0, 0, 0, 1]),
    ([2, 0, 1, 3], [0, 0, 0, 1]),
    ([2, 1, 0, 3], [0, 0, 0, 1]),
    ([0, 1, 0, 3], [0, 1, 0, 0]),

]
print("Тестовые данные:")
for input_stat, correct_predict in test_data:
    print("Рынок: {} Инвестор: {} Ожидалось: {}".format(
        str(input_stat),
        str(action[network.predict(np.array(input_stat)).argmax()]),
        str(action[np.array(correct_predict).argmax()])))
