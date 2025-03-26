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


    cutoff = 1000
    fs = mus[0].rate

    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(5, normal_cutoff, btype='low', analog=False)

    output_3_axe_nou = genereaza_strat_banda(output, (b, a))[..., newaxis]

    #output_3_axe    = np.concatenate([output_3_axe, genereaza_strat_banda(output, (b, a))[..., newaxis]], axis=2)


    cutoff = np.array([1000, 10000])
    fs = mus[0].rate

    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(5, normal_cutoff, btype='band', analog=False)

    #output_3_axe = np.concatenate([output_3_axe, genereaza_strat_banda(output, (b, a))[..., newaxis]], axis=2)
    output_3_axe_nou = np.concatenate([output_3_axe_nou, genereaza_strat_banda(output, (b, a))[..., newaxis]], axis=2)


    cutoff = 10000
    fs = mus[0].rate

    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(5, normal_cutoff, btype='high', analog=False)

    #output_3_axe = np.concatenate([output_3_axe, genereaza_strat_banda(output, (b, a))[..., newaxis]], axis=2)
    output_3_axe_nou = np.concatenate([output_3_axe_nou, genereaza_strat_banda(output, (b, a))[..., newaxis]], axis=2)

    #return output_3_axe
    return output_3_axe_nou

genereaza_tensor_din_stereo(mus[0].audio)

'''
output_file = tensor_3_axe[:, 2, 1]
print(output_file.dtype)
print(f'OUTPUT TENSOR (TEST) \n{output_file}\n\tSHAPE: {output_file.shape}')
output_file = output_file * 32767
output_file = output_file.astype(np.int16)
print(f'OUTPUT TENSOR (TEST) \n{output_file}\n\tSHAPE: {output_file.shape}')

write("output wav.wav", fs, output_file)
'''

nr_samples = mus[0].audio.shape[0]




class AudioModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_bands = torch.nn.Sequential(
            torch.nn.Linear(3, 3),
            torch.nn.Linear(3, 2),
            torch.nn.Linear(2, 1)
        )
        self.seq_channels = torch.nn.Sequential(
            torch.nn.Linear(5, 4),
            torch.nn.Linear(4, 3),
            torch.nn.Linear(3, 2),
        )
        self.seq_stems = torch.nn.Sequential(
            torch.nn.Linear(1, 2),
            torch.nn.Linear(2, 3),
            torch.nn.Linear(3, 4)
        )


    def forward(self, x):
        #print(f'forward: {x.shape}')
        x = self.seq_bands(x)
        x = torch.permute(x, (0, 2 ,1))
        x = self.seq_channels(x)
        x = torch.permute(x, (0, 2, 1))
        x = self.seq_stems(x)
        x = torch.permute(x, (2, 0, 1))
        print(x.shape)
        return x




dtype = torch.float32
device = torch.device("cpu")

model = AudioModel()

#original L1Loss
criterion = torch.nn.MSELoss(reduction='mean')
criterion.requires_grad_(True)

print(f'shape musdb {mus}')

#original 2e-5
learning_rate = 2e-2
#original era Adam aici
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9)

torch.set_grad_enabled(True)

sdr = SignalDistortionRatio

for t in range(0, 200):
    for song in range(len(mus)):
        x_true = torch.from_numpy(genereaza_tensor_din_stereo(mus[song].audio))
        x_true = x_true.to(torch.float32)
        #print(f'x_true.shape: {x_true.shape}')

        y_true = torch.from_numpy(mus[song].stems[1:, :, :])
        y_true = y_true.to(torch.float32)
        #print(f'y_true.shape: {y_true.shape}')

        y_pred = model(x_true)
        #print(f'y_pred.shape: {y_pred.shape}')



        loss = criterion(y_pred, y_true)
        #if t % 10 == 9:
        #plafoneaza pe la 0.05
        print(f't- {t}, song- {song}, mse: {loss.item()}, rmse:{math.sqrt(loss.item())}')

        y_pred_np = y_pred.detach().numpy()

        if song % 10 == 9:
            estimates = {
                'drums': y_pred_np[0, :, :],
                'bass': y_pred_np[1, :, :],
                'other': y_pred_np[2, :, :],
                'vocals': y_pred_np[3, :, :]
            }

            scores = museval.eval_mus_track(
                mus[song],
                estimates
            )

            print(scores)


        #sdr, sir, sar, perm = fast_bss_eval.bss_eval_sources(y_pred.T, y_true.T)
        #print(f'SDR: {sdr}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
