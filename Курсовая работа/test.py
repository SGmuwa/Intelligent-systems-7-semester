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

result = None
result_available = threading.Event()

if not os.path.exists(tmppath_download):
    if not os.path.exists(tmppath):
        os.mkdir(tmppath)
    os.mkdir(tmppath_download)
secret_data = [line.rstrip('\n') for line in open('api_id_hash.secret')]
(api_id, api_hash) = (int(secret_data[0]), secret_data[1])

async def download_media(message: types.Message):
    if message.file:
        print('save...')
        path = await message.download_media('{}/download'.format(tmppath), progress_callback=download_progress_callback)
        newpath = path.replace('/download', '')
        os.rename(path, newpath)
        print('File saved to', newpath)  # printed after download is done
        result = newpath
    else:
        result = None
        print('not file')
    result_available.set()
    return result

    def bytes_to_string(byte_count):
        """Converts a byte count to a string (in KB, MB...)"""
        suffix_index = 0
        while byte_count >= 1024:
            byte_count /= 1024
            suffix_index += 1
    
        return '{:.2f}{}'.format(
            byte_count, [' bytes', 'KB', 'MB', 'GB', 'TB'][suffix_index]
        )
    
    @staticmethod
    def download_progress_callback(downloaded_bytes, total_bytes):
        print_progress(
            'Downloaded', downloaded_bytes, total_bytes
        )

    @staticmethod
    def print_progress(progress_type, downloaded_bytes, total_bytes):
        print('{} {} out of {} ({:.2%})'.format(
            progress_type, bytes_to_string(downloaded_bytes),
            bytes_to_string(total_bytes), downloaded_bytes / total_bytes)
        )
def sgmuwa_cnn_bot(client: TelegramClient):
    async def handler(event: events.NewMessage.Event):
        print('sgmuwa_cnn_bot', 'got new message.', event.chat.title, event.message.text)
        path = await download_media(event.message)
        await event.respond('Saved.' if path else 'Not media.')
    client.add_event_handler(handler, events.NewMessage())

def sg_muwa(client: TelegramClient, other: TelegramClient):
    for message in client.iter_messages('vk_db'):
        print('sg_muwa', message.id, message.text, message.chat.title)
        if message.file:
            client.send_message('sgmuwa_cnn_bot', message)
            print('ok, wait...')
            while not result_available.is_set():
                print('.', end='', flush=True)
                other.catch_up()
            if result:
                yield result

print('sgmuwa_cnn_bot')
client_sgmuwa_cnn_bot = TelegramClient('sgmuwa_cnn_bot', api_id, api_hash)
client_sgmuwa_cnn_bot.connect()
print('sg_muwa')
client_sg_muwa = TelegramClient('sg_muwa', api_id, api_hash)
client_sg_muwa.connect()

def getFileIterator():
    sgmuwa_cnn_bot(client_sgmuwa_cnn_bot)
    return sg_muwa(client_sg_muwa, client_sgmuwa_cnn_bot)

def main():
    for file in getFileIterator():
        print('ok:', file)
    else:
        print('bad')

if __name__ == '__main__':
    main()
