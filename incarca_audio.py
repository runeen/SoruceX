from scipy.io.wavfile import read
import numpy as np


def incarca_sample_uri(cale_fisier : str, nr_sampleuri_per_block : int, total_samples: int):
    a = read(cale_fisier)
    a = np.array(a[1][:total_samples], dtype=np.float32)

    bucati = []
    counter = 0
    while counter + nr_sampleuri_per_block < a.__len__():
        bucati.append(a[counter: counter + nr_sampleuri_per_block])
        counter += nr_sampleuri_per_block

    return bucati, counter

