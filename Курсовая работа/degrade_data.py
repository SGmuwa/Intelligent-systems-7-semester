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

def create_good_bad_sound(filename: str):
    soundGood = pydub.AudioSegment.from_mp3(filename)
    soundGood = soundGood.set_channels(1).set_frame_rate(5000)
    soundBad = soundGood
    print('len good:', len(soundGood.raw_data), 'len bad: ', len(soundBad.raw_data))
    soundBad = soundBad.set_frame_rate(4900)
    print('len good:', len(soundGood.raw_data), 'len bad: ', len(soundBad.raw_data))
    return (soundGood, soundBad)

def generator_10sec_song(sound: pydub.AudioSegment, duration: int = 100):
    while len(sound) > duration:
        yield sound[:duration]
        sound = sound[duration:]
    yield sound

def generator_bad_and_good_sound(mp3):
    (soundGood, soundBad) = create_good_bad_sound(mp3)
    good_gen = generator_10sec_song(soundGood)
    bad_gen = generator_10sec_song(soundBad)
    for g, b in zip(good_gen, bad_gen):
        yield(b.raw_data, g.raw_data)

if __name__ == '__main__':
    import download_data
    for path in download_data.getFileIterator():
        for a in generator_bad_and_good_sound(path):
            print('len(sound[0]):', len(a[0]))
            print('len(sound[1]):', len(a[1]))
            break
        break

