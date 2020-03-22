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

import pydub
# apt-get install libav-tools libavcodec-extra

def mp3_sound(filename: str):
    sound = pydub.AudioSegment.from_mp3(filename)
    if filename[-4:] == '.mp3':
        filename = filename[:-4]
    filename += '.wav'
    sound.export(filename, format="wav")
    return filename

def generator_10sec_song(sound: pydub.AudioSegment):
    while len(song) > 10 * 1000:
        yield sound[:10 * 1000]
        sound = [10 * 1000:]
    else:
        return sound

def generator_small_and_big_wav(mp3):
    yield mp3_to_wav(mp3)

if __name__ == '__main__':
    import download_data
    for path in download_data.getFileIterator():
        for a in generator_small_and_big_wav(path):
            print(a)

