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

import sys
import soundcloud
import getpass

def main(argv):
    # create a client object with access token
    client = soundcloud.Client(access_token=getpass.getpass())

    # print out the user's username
    print(client.get('/me').username)

    # update the user's profile description
    user = client.post('/me', description='I am using the SoundCloud API!')
    print(user.description)

if __name__ == "__main__":
    main(sys.argv)