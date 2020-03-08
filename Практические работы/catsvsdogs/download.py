#!/usr/bin/python3
import os
if not os.path.exists("train"):
 if not os.path.exists("/tmp/train.zip"):
  from urllib.request import urlretrieve
  urlretrieve(input("url to zip for train from 'https://www.kaggle.com/c/dogs-vs-cats/data': "), "/tmp/train.zip")
 import zipfile
 try:
  fantasy_zip = zipfile.ZipFile('/tmp/train.zip')
 except zipfile.BadZipFile:
  with open('/tmp/train.zip', 'r') as f:
   print('bad url!', f.read())
  os.remove('/tmp/train.zip')
  exit()
 fantasy_zip.extractall('train')
 fantasy_zip.close()
