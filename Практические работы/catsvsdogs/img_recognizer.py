#!/usr/bin/python3
def recognize_img(model, classes):
 from pathlib import Path
 for path in Path().rglob('*.jpg'):
  print(path)
 img_path = input('image path or url: ')
 import re
 if re.search(r'https?://', img_path).group(0) is not None:
  from urllib.request import urlretrieve
  urlretrieve(img_path, "/tmp/.jpg")
  img_path = "/tmp/.jpg"
 from tensorflow.keras.preprocessing import image
 img = image.load_img(img_path, target_size=(28, 28), color_mode = "grayscale")
 
 # Преобразуем картинку в массив
 x = image.img_to_array(img)
 # Меняем форму массива в плоский вектор
 x = x.reshape(1, 784)
 # Инвертируем изображение
 x = 255 - x
 # Нормализуем изображение
 x /= 255
 
 """Запускаем распознавание"""
 
 prediction = model.predict(x)
 
 """Результаты распознавания"""
 
 print(prediction)
 
 prediction = np.argmax(prediction)
 print("Номер класса:", prediction)
 print("Название класса:", classes[prediction])
