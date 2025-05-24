import musdb
import numpy
from scipy.io.wavfile import write

mus = musdb.DB(subsets='test')

songs_nr = (7, 13, 17, 20, 22, 25, 26, 33, 36, 41, 42)
for song_nr in songs_nr:
    song = mus[song_nr].audio

    write(f'songs/song{song_nr}.wav', 44100, (song * 32767).astype(numpy.int16))