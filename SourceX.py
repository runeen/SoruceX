import time

import torch
import math
import random
import numpy as np
from torchmetrics.audio import SignalDistortionRatio
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
torch.backends.cudnn.deterministic = True
print(np.__version__)
print(torch.__version__)
import musdb
import tqdm
import museval
import torchmetrics
from scipy.io.wavfile import write

from scipy.signal import butter, filtfilt
import scipy.io
from torch import newaxis


mus = musdb.DB(download=True)

def center_trim(tensor, reference):
    #aproape la fel ca in demucs...

    reference = reference.size(-1)
    delta = tensor.size(-1) - reference
    tensor = tensor[..., delta // 2 : -(delta - delta // 2)]


def genereaza_strat_banda(tensor, filter):
    y = filtfilt(filter[0], filter[1], tensor, 0)
    #print(f'Y: {y}')
    return y

def genereaza_tensor_din_stereo(tensor):
    '''
    mono = np.average(tensor, 1).reshape(-1, 1)
    #print(f'MONO SHAPE: {mono.shape}')

    output = np.hstack([tensor, mono])
    #print(f'TENSOR: \n{output}\n\tSHAPE: {output.shape}')

    separated = tensor - mono

    output = np.hstack([output, separated])
    #print(f'TENSOR: \n{output}\n\tSHAPE: {output.shape}')
    '''
    output = tensor
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
    #print(f'{output_3_axe_nou.shape}')
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




class AudioModel(torch.nn.Module):
    def __init__(self, tanh_like, mish_like, relu_like):
        super().__init__()

        #encoders:
        #layer 1
        self.enc_f_1 = torch.nn.Sequential(torch.nn.Conv2d(3, 30, (20, 2), padding='same', padding_mode='circular'), tanh_like)
        self.enc_dwn_1 =  torch.nn.Sequential(torch.nn.Conv1d(30, 30, 2, stride=2), tanh_like)

        #layer 2
        self.enc_f_2 = torch.nn.Sequential(torch.nn.Conv2d(30, 60, (20, 2), padding='same', padding_mode='circular'), mish_like)
        self.enc_dwn_2 = torch.nn.Sequential(torch.nn.Conv1d(60, 60, 2, stride=2), mish_like)

        #layer 3
        self.enc_f_3 = torch.nn.Sequential(torch.nn.Conv2d(60, 100, (20, 2), padding='same', padding_mode='circular'), mish_like)
        self.enc_dwn_3 = torch.nn.Sequential(torch.nn.Conv1d(100, 100, 3, stride=3), mish_like)

        #layer 4
        self.enc_f_4 = torch.nn.Sequential(torch.nn.Conv2d(100, 300, (20, 2), padding='same', padding_mode='circular'), mish_like)
        self.enc_dwn_4 = torch.nn.Sequential(torch.nn.Conv1d(300, 300, 4, stride=4), relu_like)


        #layer 5
        self.enc_f_5 = torch.nn.Sequential(torch.nn.Conv2d(300, 600, (20, 2), padding='same', padding_mode='circular'), relu_like)
        self.enc_dwn_5 = torch.nn.Sequential(torch.nn.Conv1d(600, 600, 4, stride=4), relu_like)


        ## Nu merge asa draga mai rescrie!
        #layer 5
        self.dec_f1_5 = torch.nn.Sequential(torch.nn.Conv2d(600, 600, (20, 2), padding='same', padding_mode='circular'), relu_like)
        self.dec_ups_5 = torch.nn.Sequential(torch.nn.ConvTranspose1d(600, 600, 4, 4), relu_like)
        self.dec_f2_5 = torch.nn.Sequential(torch.nn.Conv2d(1200, 600, (20, 2), padding='same', padding_mode='circular'), torch.nn.GLU(0))

        #layer 4
        self.dec_f1_4 = torch.nn.Sequential(torch.nn.Conv2d(300, 300, (20, 2), padding='same', padding_mode='circular'), relu_like)
        self.dec_ups_4 = torch.nn.Sequential(torch.nn.ConvTranspose1d(300, 300, 4, 4), relu_like)
        self.dec_f2_4 = torch.nn.Sequential(torch.nn.Conv2d(600, 200, (20, 2), padding='same', padding_mode='circular'), torch.nn.GLU(0))


        #layer 3
        self.dec_f1_3 = torch.nn.Sequential(torch.nn.Conv2d(100, 100, (20, 2), padding='same', padding_mode='circular'), relu_like)
        self.dec_ups_3 = torch.nn.Sequential(torch.nn.ConvTranspose1d(100, 100, 3, 3), relu_like)
        self.dec_f2_3 = torch.nn.Sequential(torch.nn.Conv2d(200, 120, (20, 2), padding='same', padding_mode='circular'), torch.nn.GLU(0))


        #layer 2
        self.dec_f1_2 = torch.nn.Sequential(torch.nn.Conv2d(60, 60, (20, 2), padding='same', padding_mode='circular'), relu_like)
        self.dec_ups_2 = torch.nn.Sequential(torch.nn.ConvTranspose1d(60, 60, 2, 2), relu_like)
        self.dec_f2_2 = torch.nn.Sequential(torch.nn.Conv2d(120, 60, (20, 2), padding='same', padding_mode='circular'), torch.nn.GLU(0))

        #layer 1
        self.dec_f1_1 = torch.nn.Sequential(torch.nn.Conv2d(30, 30, (20, 2), padding='same', padding_mode='circular'), tanh_like)
        self.dec_ups_1 = torch.nn.Sequential(torch.nn.ConvTranspose1d(30, 30, 2, 2), tanh_like)
        self.dec_f2_1 = torch.nn.Sequential(torch.nn.Conv2d(60, 8, (20, 2), padding='same', padding_mode='circular'), torch.nn.GLU(0))


    def pad_x(self, x :torch.Tensor, stride: int):
        #print(x.shape)
        if (x.shape[2]) % stride != 0:
            delta = stride - (x.shape[2] % stride)
            x = torch.cat([x, torch.zeros((x.shape[0], x.shape[1], delta + stride))], dim=2)
        else:
            x = torch.cat([x, torch.zeros((x.shape[0], x.shape[1], stride))], dim=2)

        return x

    def forward(self, x : torch.Tensor):
        skip = []
        #x intra ca (T, C, B)
        #print(x.shape)


        # enc layer 1: -------------------------------------------------------------------
        #pentru Conv2d vrem(B, T, C) (Permutatie (T, C, B) -> (B, T, C))
        x = x.permute(2, 0, 1)
        x = self.enc_f_1(x)

        #pentru Conv1d vrem(C, B, T) (Permutatie (B, T, C) -> (C, B, T))
        x = x.permute(2, 0, 1)
        skip.append(x.clone())
        x = self.pad_x(x, 2)
        #print(f'x in = {x.shape}')
        x = self.enc_dwn_1(x)
        #print(f'x in = {x.shape}')

        #pentru urmatorul Conv2d vrem (B, T, C) (Permutatie (C, B, T) -> (B, T, C)))
        x = x.permute(1, 2, 0)



        # enc layer 2: -------------------------------------------------------------------
        x = self.enc_f_2(x)

        # pentru Conv1d vrem(C, B, T) (Permutatie (B, T, C) -> (C, B, T))
        x = x.permute(2, 0, 1)
        skip.append(x.clone())
        x = self.pad_x(x, 2)
        #print(f'x in = {x.shape}')
        x = self.enc_dwn_2(x)
        #print(f'x out = {x.shape}')

        # pentru urmatorul Conv2d vrem (B, T, C) (Permutatie (C, B, T) -> (B, T, C)))
        x = x.permute(1, 2, 0)



        # enc layer 3: -------------------------------------------------------------------
        x = self.enc_f_3(x)

        # pentru Conv1d vrem(C, B, T) (Permutatie (B, T, C) -> (C, B, T))
        x = x.permute(2, 0, 1)
        skip.append(x.clone())
        x = self.pad_x(x, 3)
        #print(f'x in = {x.shape}')
        x = self.enc_dwn_3(x)
        #print(f'x out = {x.shape}')

        # pentru urmatorul Conv2d vrem (B, T, C) (Permutatie (C, B, T) -> (B, T, C)))
        x = x.permute(1, 2, 0)



        # enc layer 4: -------------------------------------------------------------------
        x = self.enc_f_4(x)

        # pentru Conv1d vrem(C, B, T) (Permutatie (B, T, C) -> (C, B, T))
        x = x.permute(2, 0, 1)
        skip.append(x.clone())
        x = self.pad_x(x, 4)
        #print(f'x in = {x.shape}')
        x = self.enc_dwn_4(x)
        #print(f'x out = {x.shape}')

        # pentru urmatorul Conv2d vrem (B, T, C) (Permutatie (C, B, T) -> (B, T, C)))
        x = x.permute(1, 2, 0)



        # enc layer 5: -------------------------------------------------------------------
        x = self.enc_f_5(x)

        # pentru Conv1d vrem(C, B, T) (Permutatie (B, T, C) -> (C, B, T))
        x = x.permute(2, 0, 1)
        skip.append(x.clone())
        x = self.pad_x(x, 4)
        #print(f'x in = {x.shape}')
        x = self.enc_dwn_5(x)
        #print(f'x out = {x.shape}')

        # pentru urmatorul Conv2d vrem (B, T, C) (Permutatie (C, B, T) -> (B, T, C)))
        x = x.permute(1, 2, 0)




        #dec layer 5: -----------------------------------------------------------------
        x = self.dec_f1_5(x)
        # pentru Conv1d vrem(C, B, T) (Permutatie (B, T, C) -> (C, B, T))
        x = x.permute(2, 0, 1)
        #print(f'x in = {x.shape}')
        x = self.dec_ups_5(x)
        #print(f'x out = {x.shape}')
        skip_tensor = skip.pop(-1)
        x = torch.cat([x[:, :, :skip_tensor.shape[2]], skip_tensor], dim = 1)
        # pentru Conv2d vrem (B, T, C) (Permutatie (C, B, T) -> (B, T, C)))
        x = x.permute(1, 2, 0)

        x = self.dec_f2_5(x)


        #dec layer 4: -----------------------------------------------------------------
        x = self.dec_f1_4(x)
        # pentru Conv1d vrem(C, B, T) (Permutatie (B, T, C) -> (C, B, T))
        x = x.permute(2, 0, 1)

        #print(f'x in = {x.shape}')
        x = self.dec_ups_4(x)
        #print(f'x out = {x.shape}')
        #print(x.shape)
        skip_tensor = skip.pop(-1)
        x = torch.cat([x[:, :, :skip_tensor.shape[2]], skip_tensor], dim = 1)
        # pentru Conv2d vrem (B, T, C) (Permutatie (C, B, T) -> (B, T, C)))
        x = x.permute(1, 2, 0)

        x = self.dec_f2_4(x)


        #dec layer 3: -----------------------------------------------------------------
        x = self.dec_f1_3(x)
        # pentru Conv1d vrem(C, B, T) (Permutatie (B, T, C) -> (C, B, T))
        x = x.permute(2, 0, 1)

        x = self.dec_ups_3(x)
        #print(x.shape)
        skip_tensor = skip.pop(-1)
        x = torch.cat([x[:, :, :skip_tensor.shape[2]], skip_tensor], dim = 1)
        # pentru Conv2d vrem (B, T, C) (Permutatie (C, B, T) -> (B, T, C)))
        x = x.permute(1, 2, 0)

        x = self.dec_f2_3(x)


        #dec layer 2: -----------------------------------------------------------------
        x = self.dec_f1_2(x)
        # pentru Conv1d vrem(C, B, T) (Permutatie (B, T, C) -> (C, B, T))
        x = x.permute(2, 0, 1)

        x = self.dec_ups_2(x)
        #print(x.shape)
        skip_tensor = skip.pop(-1)
        x = torch.cat([x[:, :, :skip_tensor.shape[2]], skip_tensor], dim = 1)
        # pentru Conv2d vrem (B, T, C) (Permutatie (C, B, T) -> (B, T, C)))
        x = x.permute(1, 2, 0)

        x = self.dec_f2_2(x)


        #dec layer 1: -----------------------------------------------------------------
        x = self.dec_f1_1(x)
        # pentru Conv1d vrem(C, B, T) (Permutatie (B, T, C) -> (C, B, T))
        x = x.permute(2, 0, 1)

        x = self.dec_ups_1(x)
        #print(x.shape)
        skip_tensor = skip.pop(-1)
        x = torch.cat([x[:, :, :skip_tensor.shape[2]], skip_tensor], dim = 1)
        # pentru Conv2d vrem (B, T, C) (Permutatie (C, B, T) -> (B, T, C)))
        x = x.permute(1, 2, 0)

        x = self.dec_f2_1(x)

        return x




dtype = torch.float32
device = torch.device("cuda")
print(torch.cuda.is_available())
torch.set_default_device("cuda")

model = AudioModel(torch.nn.Tanh(), torch.nn.Mish(), torch.nn.ReLU())

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

print(mus[0].audio)

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
        #y_pred = torch.cat((y_pred, (audio_original[:, :] - torch.sum(y_pred, dim = 0))[newaxis, ...]), dim = 0)

        loss = criterion(y_pred, y_true)
        if song % 10 == 9:
            t1 = time.perf_counter()
            print(f'dt = {t1 - t0}')
            t0 = time.perf_counter()
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

            try:
                write(f'istorie antrenari/azi/original.wav', 44100, (mus[song].audio * 32767).astype(np.int16))
                write(f'istorie antrenari/azi/drums.wav', 44100, (y_pred_np[0, :, :] * 32767).astype(np.int16))
                write(f'istorie antrenari/azi/bass.wav', 44100, (y_pred_np[1, :, :] * 32767).astype(np.int16))
                write(f'istorie antrenari/azi/vocals.wav', 44100, (y_pred_np[2, :, :] * 32767).astype(np.int16))
                write(f'istorie antrenari/azi/other.wav', 44100, (y_pred_np[3, :, :] * 32767).astype(np.int16))
            except:
                print("bruh")
            try:
                scores = museval.eval_mus_track(
                    mus[song],
                    estimates
                )
                print(scores)


                # in mus[0].stems: 1 = drums, 2 = bass, 3 = other, 4 = vocals
                # in y_true/y_pred: 0 = drums, 1 = bass, 2 = vocals, 3 = other


            except:
                print("problema cu scorurile... womp womp!")


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
