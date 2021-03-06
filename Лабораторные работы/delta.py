﻿#!/usr/bin/python3
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
# Алгоритм обучения Розенблатта (дельта-правило)
# 1. Обучение с учителем.
# 2. Полученная ошибка на выходе распространяется назад.
# 3. Критерий выхода — достаточная обученность.


class Neuron:
    def __init__(self):
        self.inputs = {}
        self.value = 0

    def setValue(self, value):
        self.value = value

    def addInput(self, Neuron, w=0):
        self.inputs[Neuron] = w

    def get(self):
        if len(self.inputs) == 0:
            return self.value
        self.value = 0
        for n, w in self.inputs.items():
            self.value += n.get() * w
        return 1 if self.value > 0 else -1

    def getBool(self):
        return self.get() > 0

    def modificationEducation(self, toAddW):
        for neuron in self.inputs.keys():
            self.inputs[neuron] += toAddW * neuron.get()
            neuron.modificationEducation(toAddW)


answers = {
    'A':
        "⬜⬛⬛⬛⬜" +
        "⬛⬜⬜⬛⬜" +
        "⬛⬜⬜⬛⬜" +
        "⬛⬛⬛⬛⬜" +
        "⬛⬜⬜⬛⬜",
    'B':
        "⬛⬛⬛⬜⬜" +
        "⬛⬜⬜⬛⬜" +
        "⬛⬛⬛⬜⬜" +
        "⬛⬜⬜⬛⬜" +
        "⬛⬛⬛⬜⬜",
    'C':
        "⬜⬛⬛⬜⬜" +
        "⬛⬜⬜⬛⬜" +
        "⬛⬜⬜⬜⬜" +
        "⬛⬜⬜⬛⬜" +
        "⬜⬛⬛⬜⬜",
    'D':
        "⬛⬛⬛⬜⬜" +
        "⬛⬜⬜⬛⬜" +
        "⬛⬜⬜⬛⬜" +
        "⬛⬜⬜⬛⬜" +
        "⬛⬛⬛⬜⬜",
    'E':
        "⬛⬛⬛⬛⬜" +
        "⬛⬜⬜⬜⬜" +
        "⬛⬛⬛⬛⬜" +
        "⬛⬜⬜⬜⬜" +
        "⬛⬛⬛⬛⬜",
    'F':
        "⬛⬛⬛⬛⬜" +
        "⬛⬜⬜⬜⬜" +
        "⬛⬛⬛⬜⬜" +
        "⬛⬜⬜⬜⬜" +
        "⬛⬜⬜⬜⬜",
    'G':
        "⬜⬛⬛⬛⬜" +
        "⬛⬜⬜⬜⬜" +
        "⬛⬜⬛⬛⬜" +
        "⬛⬜⬜⬛⬜" +
        "⬜⬛⬛⬜⬜",
    'H':
        "⬛⬜⬜⬛⬜" +
        "⬛⬜⬜⬛⬜" +
        "⬛⬛⬛⬛⬜" +
        "⬛⬜⬜⬛⬜" +
        "⬛⬜⬜⬛⬜",
    'I':
        "⬛⬛⬛⬜⬜" +
        "⬜⬛⬜⬜⬜" +
        "⬜⬛⬜⬜⬜" +
        "⬜⬛⬜⬜⬜" +
        "⬛⬛⬛⬜⬜",
    'J':
        "⬜⬛⬛⬛⬜" +
        "⬜⬜⬜⬛⬜" +
        "⬜⬜⬜⬛⬜" +
        "⬛⬜⬜⬛⬜" +
        "⬜⬛⬛⬜⬜",
    'K':
        "⬛⬜⬜⬛⬜" +
        "⬛⬜⬛⬜⬜" +
        "⬛⬛⬜⬜⬜" +
        "⬛⬜⬛⬜⬜" +
        "⬛⬜⬜⬛⬜",
    'L':
        "⬛⬜⬜⬜⬜" +
        "⬛⬜⬜⬜⬜" +
        "⬛⬜⬜⬜⬜" +
        "⬛⬜⬜⬜⬜" +
        "⬛⬛⬛⬛⬜",
    'M':
        "⬛⬜⬜⬜⬛" +
        "⬛⬛⬜⬛⬛" +
        "⬛⬜⬛⬜⬛" +
        "⬛⬜⬜⬜⬛" +
        "⬛⬜⬜⬜⬛",
    'N':
        "⬛⬜⬜⬛⬜" +
        "⬛⬛⬜⬛⬜" +
        "⬛⬜⬛⬛⬜" +
        "⬛⬜⬜⬛⬜" +
        "⬛⬜⬜⬛⬜",
    'O':
        "⬜⬛⬛⬛⬜" +
        "⬛⬜⬜⬛⬜" +
        "⬛⬜⬜⬛⬜" +
        "⬛⬜⬜⬛⬜" +
        "⬜⬛⬛⬜⬜",
    'P':
        "⬛⬛⬛⬜⬜" +
        "⬛⬜⬜⬛⬜" +
        "⬛⬛⬛⬜⬜" +
        "⬛⬜⬜⬜⬜" +
        "⬛⬜⬜⬜⬜",
    'Q':
        "⬜⬛⬛⬜⬜" +
        "⬛⬜⬜⬛⬜" +
        "⬛⬜⬜⬛⬜" +
        "⬛⬜⬜⬛⬜" +
        "⬜⬛⬛⬛⬛",
    'R':
        "⬛⬛⬛⬜⬜" +
        "⬛⬜⬜⬛⬜" +
        "⬛⬛⬛⬜⬜" +
        "⬛⬜⬛⬜⬜" +
        "⬛⬜⬜⬛⬜",
    'S':
        "⬜⬛⬛⬛⬜" +
        "⬛⬜⬜⬜⬜" +
        "⬜⬛⬛⬜⬜" +
        "⬜⬜⬜⬛⬜" +
        "⬛⬛⬛⬜⬜",
    'T':
        "⬛⬛⬛⬛⬛" +
        "⬜⬜⬛⬜⬜" +
        "⬜⬜⬛⬜⬜" +
        "⬜⬜⬛⬜⬜" +
        "⬜⬜⬛⬜⬜",
    'U':
        "⬛⬜⬜⬛⬜" +
        "⬛⬜⬜⬛⬜" +
        "⬛⬜⬜⬛⬜" +
        "⬛⬜⬜⬛⬜" +
        "⬜⬛⬛⬜⬜",
    'V':
        "⬛⬜⬜⬛⬜" +
        "⬛⬜⬜⬛⬜" +
        "⬛⬜⬜⬛⬜" +
        "⬜⬛⬜⬛⬜" +
        "⬜⬜⬛⬜⬜",
    'W':
        "⬛⬜⬛⬜⬛" +
        "⬛⬜⬛⬜⬛" +
        "⬛⬜⬛⬜⬛" +
        "⬛⬜⬛⬜⬛" +
        "⬜⬛⬜⬛⬜",
    'X':
        "⬛⬜⬜⬜⬛" +
        "⬜⬛⬜⬛⬜" +
        "⬜⬜⬛⬜⬜" +
        "⬜⬛⬜⬛⬜" +
        "⬛⬜⬜⬜⬛",
    'Y':
        "⬛⬜⬜⬜⬛" +
        "⬜⬛⬜⬛⬜" +
        "⬜⬜⬛⬜⬜" +
        "⬜⬜⬛⬜⬜" +
        "⬜⬜⬛⬜⬜",
    'Z':
        "⬛⬛⬛⬛⬜" +
        "⬜⬜⬜⬛⬜" +
        "⬜⬛⬛⬜⬜" +
        "⬛⬜⬜⬜⬜" +
        "⬛⬛⬛⬛⬜"  # ,
    # '?' :
    #     "⬜⬜⬜⬜⬜" +
    #     "⬜⬜⬜⬜⬜" +
    #     "⬜⬜⬜⬜⬜" +
    #     "⬜⬜⬜⬜⬜" +
    #     "⬜⬜⬜⬜⬜",
}


def convertStringToIntArray(string):
    return [(1 if s == '⬛' else -1) for s in string]

# Конвектирует string в Array<int>.


def convertInputFromStringToIntArrays(inputDataDictionary):
    outputData = {}
    for key, string in inputDataDictionary.items():
        outputData[key] = convertStringToIntArray(string)
    return outputData


answers = convertInputFromStringToIntArrays(answers)

neurons_input = [Neuron() for _ in range(25)]

neurons_output = {}
for char, array in answers.items():
    neurons_output[char] = Neuron()
    for neuron_input in neurons_input:
        neurons_output[char].addInput(neuron_input)

# image: Array<int>
# inputNeurons: Array<Neuron>


def setImageToInput(inputNeurons, image):
    for (n, i) in zip(inputNeurons, image):
        n.setValue(i)

# Отправляет в сеть образ, возвращает ключи выходных нейронов, которые активны.
# inputNeurons: Array<Neuron>
# outputNeurons: Dictionary<char, Neuron>
# inputImage: Array<int>


def getAnswerFromNet(inputNeurons, outputNeurons, inputImage):
    setImageToInput(inputNeurons, inputImage)
    outputKeys = set()
    for key, outN in outputNeurons.items():
        if outN.getBool():
            outputKeys.add(key)
    return outputKeys


print(getAnswerFromNet(neurons_input, neurons_output, answers['A']))


def education(neurons_input, neurons_output, answers, speed, accuracyNeed):
    good = 0
    countAll = len(answers) * len(neurons_output)
    needLastContinue = True
    while good/float(countAll) < accuracyNeed or needLastContinue:
        needLastContinue = good/float(countAll) < accuracyNeed
        good = 0
        for keyAnswer, arrayAnswer in answers.items():
            setImageToInput(neurons_input, arrayAnswer)
            for keyNeuronOutput, neuronOutput in neurons_output.items():
                isGoodChar = (1 if keyAnswer == keyNeuronOutput else -1)
                e = isGoodChar - neuronOutput.get()
                if e == 0:
                    good += 1
                else:
                    neuronOutput.modificationEducation(speed * e)
        print(good/float(countAll), 'good', good, 'countAll', countAll)


education(neurons_input, neurons_output, answers, 0.0000001, 0.99)
for key, answerArray in answers.items():
    outputNet = getAnswerFromNet(neurons_input, neurons_output, answerArray)
    print('char:', key, 'net:', outputNet)

print(getAnswerFromNet(neurons_input, neurons_output, convertStringToIntArray(
    "⬜⬛⬛⬛⬜" +
    "⬛⬜⬜⬛⬜" +
    "⬛⬜⬜⬛⬜" +
    "⬛⬛⬜⬛⬜" +
    "⬛⬜⬜⬛⬜"
)))
