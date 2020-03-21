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
