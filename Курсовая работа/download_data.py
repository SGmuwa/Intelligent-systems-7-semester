#!/usr/bin/python3
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

from telethon.sync import TelegramClient
from telethon import events
from telethon import types
import threading
import tempfile
import asyncio
import os

tmppath = '{}/{}'.format(tempfile.gettempdir(), 'vk_db')
tmppath_download = '{}/{}'.format(tmppath, 'download')
media = []

if not os.path.exists(tmppath_download):
    if not os.path.exists(tmppath):
        os.mkdir(tmppath)
    os.mkdir(tmppath_download)
secret_data = [line.rstrip('\n') for line in open('api_id_hash.secret')]
(api_id, api_hash) = (int(secret_data[0]), secret_data[1])

def download_media(client: TelegramClient, message: types.Message):
    def bytes_to_string(byte_count):
        """Converts a byte count to a string (in KB, MB...)"""
        suffix_index = 0
        while byte_count >= 1024:
            byte_count /= 1024
            suffix_index += 1

        return '{:.2f}{}'.format(
            byte_count, [' bytes', 'KB', 'MB', 'GB', 'TB'][suffix_index]
        )
    
    def download_progress_callback(downloaded_bytes, total_bytes):
        print_progress(
            'Downloaded', downloaded_bytes, total_bytes
        )

    def print_progress(progress_type, downloaded_bytes, total_bytes):
        print('\r{} {} out of {} ({:.2%})  '.format(
            progress_type, bytes_to_string(downloaded_bytes),
            bytes_to_string(total_bytes), downloaded_bytes / total_bytes), end='', flush=True
        )
    
    if message.file:
        print('save...')
        path = message.download_media('{}/download'.format(tmppath), progress_callback=download_progress_callback)
        newpath = path.replace('/download', '')
        os.rename(path, newpath)
        print('\nFile saved to', newpath)  # printed after download is done
        return newpath
    else:
        print('not file')
        return None

def getFileIterator():
    print('sg_muwa')
    with TelegramClient('sg_muwa', api_id, api_hash) as client_sg_muwa:
        for message in client_sg_muwa.iter_messages('vk_db'):
            print('sg_muwa', message.id, message.text, message.chat.title)
            if message.file:
                path = download_media(client_sg_muwa, message)
                if path is not None:
                    yield path
                    os.remove(path)

def main():
    for file in getFileIterator():
        print('ok:', file)
    else:
        print('bad')

if __name__ == '__main__':
    main()
