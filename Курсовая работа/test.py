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

from telethon.sync import TelegramClient, events
secret_data = [line.rstrip('\n') for line in open('api_id_hash.secret')]
api_id = int(secret_data[0])
api_hash = secret_data[1]
with TelegramClient('sg_muwa', api_id, api_hash) as client:
    print(client.get_me().stringify())
    client.send_message('sg_muwa', 'Hello! Talking to you from Telethon')

    @client.on(events.NewMessage(pattern='(?i)hi|hello'))
    async def handler(event):
        await event.respond('Hey!')

    client.run_until_disconnected()
