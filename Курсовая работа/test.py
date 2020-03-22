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
import os
import tempfile

tmppath = '{}/{}'.format(tempfile.gettempdir(), 'vk_db')
tmppath_download = '{}/{}'.format(tmppath, 'download')

async def download_media(message):
    if message.file:
        path = await message.download_media('{}/download'.format(tmppath))
        newpath = path.replace('/download', '')
        os.rename(path, newpath)
        print('File saved to', newpath)  # printed after download is done
        return True
    else:
        return False

def main(client: TelegramClient):

    #async for message in client.iter_messages('sg_muwa'):
    #    print(message.id, message.text)
    #    # You can download media from messages, too!
    #    # The method will return the path where the file was saved.
    #    if message.file:
    #        path = await message.download_media('{}/download'.format(tmppath))
    #        newpath = path.replace('/download', '')
    #        os.rename(path, newpath)
    #        print('File saved to', newpath)  # printed after download is done
    @client.on(events.NewMessage())
    async def handler(event: events.NewMessage.Event):
        await event.respond('Saved.' if await download_media(event.message) else 'Not media.')
    

    client.run_until_disconnected()


if __name__ == '__main__':
    if not os.path.exists(tmppath_download):
        if not os.path.exists(tmppath):
            os.mkdir(tmppath)
        os.mkdir(tmppath_download)
    
    secret_data = [line.rstrip('\n') for line in open('api_id_hash.secret')]
    (session_name, api_id, api_hash) = (secret_data[0], int(secret_data[1]), secret_data[2])
    with TelegramClient(session_name, api_id, api_hash) as client:
        main(client)
