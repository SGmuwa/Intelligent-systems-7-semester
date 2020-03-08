#!/usr/bin/python3
import os
if not os.path.exists("train"):
 if not os.path.exists("/tmp/train.zip"):
  from urllib.request import urlretrieve
  urlretrieve(input("url to zip for train: "), "/tmp/train.zip")
 import zipfile
 fantasy_zip = zipfile.ZipFile('train.zip')
 fantasy_zip.extractall('train')
 fantasy_zip.close()
