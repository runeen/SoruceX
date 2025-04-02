import time

import torch
import math
import random
import numpy as np
from torchmetrics.audio import SignalDistortionRatio
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

print(np.__version__)
import musdb
import tqdm
import museval
import torchmetrics
from scipy.io.wavfile import write

from scipy.signal import butter, filtfilt
import scipy.io
from torch import newaxis

import incarca_audio

mus = musdb.DB(download=True)


def genereaza_strat_banda(tensor, filter):
    y = filtfilt(filter[0], filter[1], tensor, 0)
    #print(f'Y: {y}')
    return y

def genereaza_tensor_din_stereo(tensor):
    mono = np.average(tensor, 1).reshape(-1, 1)
    #print(f'MONO SHAPE: {mono.shape}')

    output = np.hstack([tensor, mono])
    #print(f'TENSOR: \n{output}\n\tSHAPE: {output.shape}')

    separated = tensor - mono

    output = np.hstack([output, separated])
    #print(f'TENSOR: \n{output}\n\tSHAPE: {output.shape}')

    output_3_axe = output[..., np.newaxis]


    cutoff = 400
    fs = mus[0].rate

    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(5, normal_cutoff, btype='low', analog=False)

    output_3_axe_nou = genereaza_strat_banda(output, (b, a))[..., newaxis]


    cutoff = np.array([400, 1900])
    fs = mus[0].rate
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(5, normal_cutoff, btype='band', analog=False)
    output_3_axe_nou = np.concatenate([output_3_axe_nou, genereaza_strat_banda(output, (b, a))[..., newaxis]], axis=2)

    cutoff = 1900
    fs = mus[0].rate
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(5, normal_cutoff, btype='high', analog=False)
    output_3_axe_nou = np.concatenate([output_3_axe_nou, genereaza_strat_banda(output, (b, a))[..., newaxis]], axis=2)




    #return output_3_axe
    print(f'{output_3_axe_nou.shape}')
    return output_3_axe_nou

def apply_high_pass(tensor):
    tensor = tensor.detach().numpy()
    cutoff = 10
    fs = mus[0].rate

    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(3, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, tensor, 1)
    y = torch.tensor(y.copy(), requires_grad=True, dtype=torch.float32)
    return y


genereaza_tensor_din_stereo(mus[0].audio)


nr_samples = mus[0].audio.shape[0]




class AudioModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        #am scos 0.001 la vocals (finally) [a stat sa gateasca cam 40 de min]
        #drums si bass sunt in urma pt ca se topesc intre ele la frecventele joase?
        #   (conv se fac separat pe benzi (kernel_size[1] = 1) deci nu luam in considerare
        #   infomratiile stereo (macar nu relatia dintre ele))


        self.enc_benzi = torch.nn.Sequential(
            torch.nn.Conv2d(3, 10, (20, 5), padding='same'),
            torch.nn.Conv2d(10, 15, (20, 5), padding='same'),
            torch.nn.Conv2d(15, 20, (20, 5), padding='same'),
            torch.nn.Conv2d(20, 60, (20, 5), padding='same')
        )

        #am adaugat kernele de 5 pe dim canale... posibil sa nu ajunga nicaieri si daca asta e cazul
        #atunci scoate idk

        # gandestete la ce padding mode sa folosesti (mai ales pt stereo)
        self.dec_stem = torch.nn.Sequential(
            torch.nn.Conv2d(60, 20, (20, 5), padding='same'),
            torch.nn.Conv2d(20, 15, (20, 5), padding='same'),
            torch.nn.Conv2d(15, 10, (20, 5), padding='same'),
            torch.nn.Conv2d(10, 3, (20, 5), padding='same'),
        )


    def forward(self, x : torch.Tensor):
        x = x.permute(2, 0, 1)

        x = self.enc_benzi(x)

        x = self.dec_stem(x)

        return x[:, :, :2]




dtype = torch.float32
device = torch.device("cuda")
print(torch.cuda.is_available())
torch.set_default_device("cuda")

model = AudioModel()

criterion = torch.nn.MSELoss(reduction='mean')
criterion.requires_grad_(True)

print(f'shape musdb {mus}')

#original 2e-5
learning_rate = 2e-4
#original era Adam aici
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9)

torch.set_grad_enabled(True)

sdr = SignalDistortionRatio

t0 = time.perf_counter()
for t in range(0, 1000):
    for song in range(len(mus)):

        audio_original = mus[song].audio
        x_true = torch.from_numpy(genereaza_tensor_din_stereo(audio_original))
        audio_original = torch.from_numpy(audio_original).to(device= 'cuda', dtype=torch.float32)

        x_true = x_true.to(torch.float32)
        x_true = x_true.to(device = "cuda")

        # in mus[0].stems: 1 = drums, 2 = bass, 3 = other, 4 = vocals
        # in y_true/y_pred: 0 = drums, 1 = bass, 2 = vocals, 3 = other
        y_true = torch.from_numpy(mus[song].stems[(1, 2, 4, 3), :, :])
        y_true = y_true.to(torch.float32)
        y_true = y_true.to(device = "cuda")

        y_pred = model(x_true)
        y_pred = torch.cat((y_pred, (audio_original[:, :] - torch.sum(y_pred, dim = 0))[newaxis, ...]), dim = 0)

        loss = criterion(y_pred, y_true)
        if song % 10 == 9:
            print(f't- {t}, song- {song}, mse: {loss.item()}, rmse:{math.sqrt(loss.item())}')


        if song % 200 == 99:
            y_pred = y_pred.to(device="cpu")
            y_pred_np = y_pred.detach().numpy()

            # in mus[0].stems: 1 = drums, 2 = bass, 3 = other, 4 = vocals
            # in y_true/y_pred: 0 = drums, 1 = bass, 2 = vocals, 3 = other
            estimates = {
                'drums': y_pred_np[0, :, :],
                'bass': y_pred_np[1, :, :],
                'vocals': y_pred_np[2, :, :],
                'other': y_pred_np[3, :, :]

            }

            scores = museval.eval_mus_track(
                mus[song],
                estimates
            )
            print(scores)


            # in mus[0].stems: 1 = drums, 2 = bass, 3 = other, 4 = vocals
            # in y_true/y_pred: 0 = drums, 1 = bass, 2 = vocals, 3 = other

            try:
                write(f'istorie antrenari/2.4.25 training/original.wav', 44100, (mus[song].audio * 32767).astype(np.int16))
                write(f'istorie antrenari/2.4.25 training/drums.wav', 44100, (y_pred_np[0, :, :] * 32767).astype(np.int16))
                write(f'istorie antrenari/2.4.25 training/bass.wav', 44100, (y_pred_np[1, :, :] * 32767).astype(np.int16))
                write(f'istorie antrenari/2.4.25 training/vocals.wav', 44100, (y_pred_np[2, :, :] * 32767).astype(np.int16))
                write(f'istorie antrenari/2.4.25 training/other.wav', 44100, (y_pred_np[3, :, :] * 32767).astype(np.int16))
            except:
                print("bruh")


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t1 = time.perf_counter()
        print(f'dt = {t1 - t0}')
        t0 = time.perf_counter()
