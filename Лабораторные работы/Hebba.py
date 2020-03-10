#!/usr/bin/env python
# coding: utf-8

# In[7]:


class Net:
    def __init__(self,weight_mass): # Массив весов включает w11,w21,...,T
        self.weight_mass = weight_mass
    def modification_w(self, image):
        for i in range(len(self.weight_mass)):
            if i==len(self.weight_mass)-1:
                self.weight_mass[i] = self.weight_mass[i] - image[1][self.ind]
            else:
                self.weight_mass[i] = self.weight_mass[i] + image[0][i] * image[1][self.ind]
    def __str__(self):
        return '{} {}'.format(self.ind, self.weight_mass)
    def threshold_function(self,net):
        return 1 if net > 0 else -1
    def NET_OUT(self,image):
        output = 0
        for point in image:
            output += self.weight_mass[i]*point
        return self.threshold_function(output)
 
#Initialization of weights by random values ​​close to 0
#10 neurons
N = 15 
list_neurons=[]
NEURON_COUNT = 10
for i in range(0,NEURON_COUNT):
    list_neurons.append(Net(i,[0]*(N+1)))
#List of image
images =[]
image0 = [
    1,1,1,1,
    -1,1,1,-1,
    1,1,-1,1,
    1,1,1,-1],[]];
images.append(image0)
image1 = [[1,-1,-1,-1,1,1,1,-1,1,-1,-1,1,1,-1,-1,-1],[]];
images.append(image1)#цифра 1, последний элемент - дополнительный параметр 
image2 = [[1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,1,-1],[]];
images.append(image2)
image3 = [[1,1,1,-1,-1,1,1,1,1,-1,-1,1,1,1,1,-1],[]];
images.append(image3)
image4 = [[1,-1,1,1,-1,1,1,1,1,-1,-1,1,1,-1,-1,-1],[]];
images.append(image4)
image5 = [[1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,1,1,-1],[]];
images.append(image5)
image6 = [[1,1,1,1,-1,-1,1,1,1,1,-1,1,1,1,1,-1],[]];
images.append(image6)
image7 = [[1,1,1,-1,-1,1,1,-1,-1,-1,1,-1,-1,-1,1,-1],[]];
images.append(image7)
image8 = [[1,1,1,1,-1,1,1,1,1,1,-1,1,1,1,1,-1],[]];
images.append(image8)
image9 = [[1,1,1,1,-1,1,1,1,1,-1,-1,1,1,1,1,-1],[]];
images.append(image9)


for i in range(len(images)):
    for j in range(NEURON_COUNT):
        if i==j:
            images[i][1].append(1)
        else:  images[i][1].append(-1)

for temp in range(6):
    for indNer in range(len(list_neurons)):
        for ob in images:
            if(list_neurons[indNer].NET_OUT(ob)!=ob[1][indNer]):#1 indNer == ind_ob, else -1
                list_neurons[indNer].modification_w(ob)
                print(list_neurons[indNer]);
for j in list_neurons:
    print("NERON")
    print(j)
    print("\n")

print(end = '\t')
for j in range(len(list_neurons)):
    print("Н: {0}".format(j),end = '\t')
print()
for j in list_neurons: # строка это образ
    print("ОБРАЗ: ", end = '\t')
    for c in images:
        print(j.NET_OUT(c),end = '\t')
    print('\n')

test_image = [[1,-1,1,-1,1,1,1,-1,1,-1,-1,1,1,-1,1,-1],[-1,1,-1,-1,-1,-1,-1,-1,-1,-1]]#больше похоже на 1
for ner in range(len(list_neurons)):
    result = list_neurons[ner].NET_OUT(test_image)
    print("Результат: ", str(ner) if result == 1 else str(False))

