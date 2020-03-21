#!/usr/bin/python3
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
def recognize_img(model, classes, target_size=None, color_mode='rgb'):
 while True:
  img_path = input('image path or url or q or f: ')
  if img_path == 'q':
   break
  if img_path == 'f':
   from pathlib import Path
   for path in Path().rglob('*.jpg'):
    print(path)
   continue
  import re
  if re.search(r'https?://', img_path) is not None:
   from urllib.request import urlretrieve
   urlretrieve(img_path, "/tmp/.jpg")
   img_path = "/tmp/.jpg"
  from tensorflow.keras.preprocessing import image
  img = image.load_img(img_path, target_size=target_size, color_mode=color_mode)
  
  # Преобразуем картинку в массив
  x = image.img_to_array(img)
  # Меняем форму массива в плоский вектор
  x = x.reshape(1, target_size[0]*target_size[1])
  # Инвертируем изображение
  x = 255 - x
  # Нормализуем изображение
  x = x / 255. * 2 - 1.
  
  print("Запускаем распознавание")
  
  prediction = model.predict(x)
  
  print("Результаты распознавания")
  
  print(prediction)
  
  import numpy as np
  prediction = np.argmax(prediction)
  print("Номер класса:", prediction)
  print("Название класса:", classes[prediction])
