#!/usr/bin/python3
# coding: utf-8

# # Подготовка данных для распознавания котов и собак на изображениях
# 
# **Источник данных** - соревнования Kaggle [Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data).
# 
# Пример подготовки данных для обучения нейронной сети на Keras. Данные разбиваются на три каталога:
# - train (данные для обучения)
# - val (данные для проверки)
# - test (данные для тестирования)
# 
# В каждом каталоге создаются по два подкаталога, в соответсвии с названиями классов: cats и dogs. 
# 
# Изображения переписваются из исходного каталога в новую структуру. По-умолчанию для обучения используется 70% изображений, для проверки - 15%, для тестрования также 15%. 
# 
# Перед запуском необходимо скачать файл train.zip с набором изображений кошек и собак с сайта соревнования Kaggle [Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data) и распаковать его.

# In[1]:


import shutil
import os


# In[2]:


# Каталог с набором данных
data_dir = '/home/andrey/work/datasets/cats_dogs/train'
# Каталог с данными для обучения
train_dir = 'train'
# Каталог с данными для проверки
val_dir = 'val'
# Каталог с данными для тестирования
test_dir = 'test'
# Часть набора данных для тестирования
test_data_portion = 0.15
# Часть набора данных для проверки
val_data_portion = 0.15
# Количество элементов данных в одном классе
nb_images = 12500


# Функция создания каталога с двумя подкаталогами по названию классов: cats и dogs

# In[3]:


def create_directory(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)
    os.makedirs(os.path.join(dir_name, "cats"))
    os.makedirs(os.path.join(dir_name, "dogs"))    


# Создание структуры каталогов для обучающего, проверочного и тестового набора данных

# In[4]:


create_directory(train_dir)
create_directory(val_dir)
create_directory(test_dir)


# Функция копирования изображений в заданный каталог. Изображения котов и собак копируются в отдельные подкаталоги

# In[5]:


def copy_images(start_index, end_index, source_dir, dest_dir):
    for i in range(start_index, end_index):
        shutil.copy2(os.path.join(source_dir, "cat." + str(i) + ".jpg"), 
                    os.path.join(dest_dir, "cats"))
        shutil.copy2(os.path.join(source_dir, "dog." + str(i) + ".jpg"), 
                   os.path.join(dest_dir, "dogs"))


# Расчет индексов наборов данных для обучения, приверки и тестирования

# In[6]:


start_val_data_idx = int(nb_images * (1 - val_data_portion - test_data_portion))
start_test_data_idx = int(nb_images * (1 - test_data_portion))
print(start_val_data_idx)
print(start_test_data_idx)               


# Копирование изображений

# In[7]:


copy_images(0, start_val_data_idx, data_dir, train_dir)
copy_images(start_val_data_idx, start_test_data_idx, data_dir, val_dir)
copy_images(start_test_data_idx, nb_images, data_dir, test_dir)


# In[ ]:




