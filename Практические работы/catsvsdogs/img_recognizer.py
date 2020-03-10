#!/usr/bin/python3
def recognize_img(model, classes, target_size=None, color_mode='rgb'):
 while True:
  from pathlib import Path
  for path in Path().rglob('*.jpg'):
   print(path)
  img_path = input('image path or url or q: ')
  if img_path == 'q':
   break
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
  import numpy as np
  x = np.expand_dims(x, axis=0)
  # Нормализуем изображение
  x /= 255.
  
  print("Запускаем распознавание")
  
  prediction = model.predict(x)
  
  print("Результаты распознавания")
  
  print(prediction)
  
  prediction = np.argmax(prediction)
  print("Номер класса:", prediction)
  print("Название класса:", classes[prediction])
