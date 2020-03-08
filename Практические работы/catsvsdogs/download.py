#!/usr/bin/python3

def get_size(path='.'):
 from pathlib import Path
 root_directory = Path(path)
 return sum(f.stat().st_size for f in root_directory.glob('**/*') if f.is_file() )

import os
if not os.path.exists("data") or get_size("data") < 545.4*1024*1024:
 if not os.path.exists("/tmp/data.zip"):
  from urllib.request import urlretrieve
  urlretrieve(input("url to zip for train from 'https://www.kaggle.com/c/dogs-vs-cats/data': "), "/tmp/data.zip")
 import zipfile
 try:
  fantasy_zip = zipfile.ZipFile('/tmp/data.zip')
 except zipfile.BadZipFile:
  with open('/tmp/data.zip', 'r') as f:
   print('bad zip!', f.read())
  os.remove('/tmp/data.zip')
  exit()
 fantasy_zip.extractall('.')
 fantasy_zip.close()
 os.rename('train', 'data')
