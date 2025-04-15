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
import augment
import gc
from tqdm import tqdm

from scipy.signal import butter, filtfilt
import scipy.io
from torch import newaxis
import sys



def center_trim(tensor, reference):
    #aproape la fel ca in demucs...

    reference = reference.size(-1)
    delta = tensor.size(-1) - reference
    tensor = tensor[..., delta // 2 : -(delta - delta // 2)]


def genereaza_strat_banda(tensor, filter):
    y = filtfilt(filter[0], filter[1], tensor, 0)
    #print(f'Y: {y}')
    return y

def genereaza_tensor_din_stereo(tensor, fs = 44100):
    '''
    mono = np.average(tensor, 1).reshape(-1, 1)
    #print(f'MONO SHAPE: {mono.shape}')

    y_bar = np.hstack([tensor, mono])
    #print(f'TENSOR: \n{y_bar}\n\tSHAPE: {y_bar.shape}')

    separated = tensor - mono

    y_bar = np.hstack([y_bar, separated])
    #print(f'TENSOR: \n{y_bar}\n\tSHAPE: {y_bar.shape}')
    '''
    output = tensor
    output_3_axe = output[..., np.newaxis]


    cutoff = 400

    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(5, normal_cutoff, btype='low', analog=False)

    output_3_axe_nou = genereaza_strat_banda(output, (b, a))[..., newaxis]


    cutoff = np.array([400, 1900])
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(5, normal_cutoff, btype='band', analog=False)
    output_3_axe_nou = np.concatenate([output_3_axe_nou, genereaza_strat_banda(output, (b, a))[..., newaxis]], axis=2)

    cutoff = 1900
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

        # --- encoders:
        #layer 1
        self.enc_f_1 = torch.nn.Sequential(torch.nn.Conv2d(3, 30, (2, 2), padding='same', padding_mode='circular'), tanh_like)
        self.enc_dwn_1 =  torch.nn.Sequential(torch.nn.Conv1d(30, 60, 4, stride=4), tanh_like)

        #layer 2
        self.enc_f_2 = torch.nn.Sequential(torch.nn.Conv2d(60, 60, (2, 2), padding='same', padding_mode='circular'), mish_like)
        self.enc_dwn_2 = torch.nn.Sequential(torch.nn.Conv1d(60, 100, 4, stride=4), mish_like)

        #layer 3
        self.enc_f_3 = torch.nn.Sequential(torch.nn.Conv2d(100, 100, (2, 2), padding='same', padding_mode='circular'), mish_like)
        self.enc_dwn_3 = torch.nn.Sequential(torch.nn.Conv1d(100, 300, 4, stride=4), mish_like)

        #layer 4
        self.enc_f_4 = torch.nn.Sequential(torch.nn.Conv2d(300, 300, (2, 2), padding='same', padding_mode='circular'), mish_like)
        self.enc_dwn_4 = torch.nn.Sequential(torch.nn.Conv1d(300, 600, 4, stride=4), relu_like)

        #layer 5
        self.enc_f_5 = torch.nn.Sequential(torch.nn.Conv2d(600, 600, (2, 2), padding='same', padding_mode='circular'), relu_like)
        self.enc_dwn_5 = torch.nn.Sequential(torch.nn.Conv1d(600, 900, 4, stride=4), relu_like)

        self.blstm = torch.nn.LSTM(900, 900, bidirectional=True, num_layers=3)
        self.blstm_linear = torch.nn.Linear(1800, 600)

        # --- decoders:
        #layer 5
        self.dec_f1_5 = torch.nn.Sequential(torch.nn.Conv2d(600, 600, (2, 2), padding='same', padding_mode='circular'), relu_like)
        self.dec_ups_5 = torch.nn.Sequential(torch.nn.ConvTranspose1d(600, 600, 4, 4), relu_like)
        self.dec_f2_5 = torch.nn.Sequential(torch.nn.Conv2d(1200, 600, (2, 2), padding='same', padding_mode='circular'), torch.nn.GLU(0))

        #layer 4
        self.dec_f1_4 = torch.nn.Sequential(torch.nn.Conv2d(300, 300, (2, 5), padding='same', padding_mode='circular'), relu_like)
        self.dec_ups_4 = torch.nn.Sequential(torch.nn.ConvTranspose1d(300, 300, 4, 4), relu_like)
        self.dec_f2_4 = torch.nn.Sequential(torch.nn.Conv2d(600, 200, (2, 2), padding='same', padding_mode='circular'), torch.nn.GLU(0))


        #layer 3
        self.dec_f1_3 = torch.nn.Sequential(torch.nn.Conv2d(100, 100, (2, 2), padding='same', padding_mode='circular'), relu_like)
        self.dec_ups_3 = torch.nn.Sequential(torch.nn.ConvTranspose1d(100, 100, 4, 4), relu_like)
        self.dec_f2_3 = torch.nn.Sequential(torch.nn.Conv2d(200, 120, (2, 2), padding='same', padding_mode='circular'), torch.nn.GLU(0))


        #layer 2
        self.dec_f1_2 = torch.nn.Sequential(torch.nn.Conv2d(60, 60, (2, 2), padding='same', padding_mode='circular'), relu_like)
        self.dec_ups_2 = torch.nn.Sequential(torch.nn.ConvTranspose1d(60, 60, 4, 4), relu_like)
        self.dec_f2_2 = torch.nn.Sequential(torch.nn.Conv2d(120, 60, (2, 2), padding='same', padding_mode = 'circular'), torch.nn.GLU(0))

        #layer 1
        self.dec_f1_1 = torch.nn.Sequential(torch.nn.Conv2d(30, 30, (2, 2), padding='same', padding_mode='circular'), tanh_like)
        self.dec_ups_1 = torch.nn.Sequential(torch.nn.ConvTranspose1d(30, 30, 4, 4), tanh_like)
        self.dec_f2_1 = torch.nn.Sequential(torch.nn.Conv2d(60, 8, (2, 2), padding='same', padding_mode='circular'), torch.nn.GLU(0))



    def pad_x(self, x :torch.Tensor, stride: int):
        #print(x.shape)
        if (x.shape[2]) % stride != 0:
            delta = stride - (x.shape[2] % stride)
            x = torch.cat([x, torch.zeros((x.shape[0], x.shape[1], delta + stride), device='cuda')], dim=2)
        else:
            x = torch.cat([x, torch.zeros((x.shape[0], x.shape[1], stride), device='cuda')], dim=2)
        return x

    def forward(self, x : torch.Tensor):
        skip = []
        #print(x.shape)
        #x intra ca (T, C, B)
        #print(x.shape)


        # enc layer 1: -------------------------------------------------------------------
        #pentru Conv2d vrem(B, T, C) (Permutatie (T, C, B) -> (B, T, C))
        x = x.permute(2, 0, 1)
        x = self.enc_f_1(x)

        #pentru Conv1d vrem(C, B, T) (Permutatie (B, T, C) -> (C, B, T))
        x = x.permute(2, 0, 1)
        #x = x.to('cpu')
        skip.append(x)
        x = self.pad_x(x, 4)
        #x = x.to('cuda')
        #print(f'x in = {x.shape}')
        x = self.enc_dwn_1(x)
        #print(f'x in = {x.shape}')

        #pentru urmatorul Conv2d vrem (B, T, C) (Permutatie (C, B, T) -> (B, T, C)))
        x = x.permute(1, 2, 0)



        # enc layer 2: -------------------------------------------------------------------
        x = self.enc_f_2(x)

        # pentru Conv1d vrem(C, B, T) (Permutatie (B, T, C) -> (C, B, T))
        x = x.permute(2, 0, 1)
        #x = x.to('cpu')
        skip.append(x)
        x = self.pad_x(x, 4)
        #x = x.to('cuda')
        #print(f'x in = {x.shape}')
        x = self.enc_dwn_2(x)
        #print(f'x out = {x.shape}')

        # pentru urmatorul Conv2d vrem (B, T, C) (Permutatie (C, B, T) -> (B, T, C)))
        x = x.permute(1, 2, 0)



        # enc layer 3: -------------------------------------------------------------------
        x = self.enc_f_3(x)

        # pentru Conv1d vrem(C, B, T) (Permutatie (B, T, C) -> (C, B, T))
        x = x.permute(2, 0, 1)
        #x = x.to('cpu')
        skip.append(x)
        x = self.pad_x(x, 4)
        #x = x.to('cuda')
        #print(f'x in = {x.shape}')
        x = self.enc_dwn_3(x)
        #print(f'x out = {x.shape}')

        # pentru urmatorul Conv2d vrem (B, T, C) (Permutatie (C, B, T) -> (B, T, C)))
        x = x.permute(1, 2, 0)


        # enc layer 4: -------------------------------------------------------------------
        x = self.enc_f_4(x)

        # pentru Conv1d vrem(C, B, T) (Permutatie (B, T, C) -> (C, B, T))
        x = x.permute(2, 0, 1)
        #x = x.to('cpu')
        skip.append(x)
        x = self.pad_x(x, 4)
        #x = x.to('cuda')
        #print(f'x in = {x.shape}')
        x = self.enc_dwn_4(x)
        #print(f'x out = {x.shape}')

        # pentru urmatorul Conv2d vrem (B, T, C) (Permutatie (C, B, T) -> (B, T, C)))
        x = x.permute(1, 2, 0)


        # enc layer 5: -------------------------------------------------------------------
        x = self.enc_f_5(x)

        # pentru Conv1d vrem(C, B, T) (Permutatie (B, T, C) -> (C, B, T))
        x = x.permute(2, 0, 1)
        #x = x.to('cpu')
        skip.append(x)
        x = self.pad_x(x, 4)
        #x = x.to('cuda')
        #print(f'x in = {x.shape}')
        x = self.enc_dwn_5(x)
        #print(f'x out = {x.shape}')

        #blstm -----------------------------------------------------------------------
        # pentru blstm vrem (T, C, B) (Permutatie (C, B, T) -> (T, C, B))
        x = x.permute(2, 0, 1)

        x = self.blstm(x)[0]

        x = self.blstm_linear(x)

        #pentru urmatorul filtru vrem (B, T, C) (Permutatie (T, C, B) -> (B, T, C))
        x = x.permute(2, 0, 1)



        #dec layer 5: -----------------------------------------------------------------
        x = self.dec_f1_5(x)
        # pentru Conv1d vrem(C, B, T) (Permutatie (B, T, C) -> (C, B, T))
        x = x.permute(2, 0, 1)
        #print(f'x in = {x.shape}')
        x = self.dec_ups_5(x)
        #print(f'x out = {x.shape}')
        #x = x.to('cpu')
        skip_tensor = skip.pop(-1)
        x = torch.cat([x[:, :, :skip_tensor.shape[2]], skip_tensor], dim = 1)
        #x = x.to('cuda')
        # pentru Conv2d vrem (B, T, C) (Permutatie (C, B, T) -> (B, T, C)))
        x = x.permute(1, 2, 0)

        gc.collect()
        x = self.dec_f2_5(x)


        #dec layer 4: -----------------------------------------------------------------
        x = self.dec_f1_4(x)
        # pentru Conv1d vrem(C, B, T) (Permutatie (B, T, C) -> (C, B, T))
        x = x.permute(2, 0, 1)

        #print(f'x in = {x.shape}')
        x = self.dec_ups_4(x)
        #print(f'x out = {x.shape}')
        #print(x.shape)
        #x = x.to('cpu')
        skip_tensor = skip.pop(-1)
        x = torch.cat([x[:, :, :skip_tensor.shape[2]], skip_tensor], dim = 1)
        #x = x.to('cuda')
        # pentru Conv2d vrem (B, T, C) (Permutatie (C, B, T) -> (B, T, C)))
        x = x.permute(1, 2, 0)

        gc.collect()
        x = self.dec_f2_4(x)


        #dec layer 3: -----------------------------------------------------------------
        x = self.dec_f1_3(x)
        # pentru Conv1d vrem(C, B, T) (Permutatie (B, T, C) -> (C, B, T))
        x = x.permute(2, 0, 1)

        x = self.dec_ups_3(x)
        #print(x.shape)
        #x = x.to('cpu')
        skip_tensor = skip.pop(-1)
        x = torch.cat([x[:, :, :skip_tensor.shape[2]], skip_tensor], dim = 1)
        #x = x.to('cuda')
        # pentru Conv2d vrem (B, T, C) (Permutatie (C, B, T) -> (B, T, C)))
        x = x.permute(1, 2, 0)

        gc.collect()
        x = self.dec_f2_3(x)


        #dec layer 2: -----------------------------------------------------------------
        x = self.dec_f1_2(x)
        # pentru Conv1d vrem(C, B, T) (Permutatie (B, T, C) -> (C, B, T))
        x = x.permute(2, 0, 1)

        x = self.dec_ups_2(x)
        #print(x.shape)
        #x = x.to('cpu')
        skip_tensor = skip.pop(-1)
        x = torch.cat([x[:, :, :skip_tensor.shape[2]], skip_tensor], dim = 1)
        #x = x.to('cuda')
        # pentru Conv2d vrem (B, T, C) (Permutatie (C, B, T) -> (B, T, C)))
        gc.collect()
        x = x.permute(1, 2, 0)
        x = self.dec_f2_2(x)



        #dec layer 1: -----------------------------------------------------------------
        x = self.dec_f1_1(x)
        # pentru Conv1d vrem(C, B, T) (Permutatie (B, T, C) -> (C, B, T))
        x = x.permute(2, 0, 1)

        gc.collect()
        x = self.dec_ups_1(x)
        #print(x.shape)
        #x = x.to('cpu')
        skip_tensor = skip.pop(-1)
        x = torch.cat([x[:, :, :skip_tensor.shape[2]], skip_tensor], dim = 1)
        #x = x.to('cuda')
        # pentru Conv2d vrem (B, T, C) (Permutatie (C, B, T) -> (B, T, C)))
        x = x.permute(1, 2, 0)

        gc.collect()
        x = self.dec_f2_1(x)

        del skip_tensor

        return x


if __name__ == '__main__':

    dtype = torch.float32
    device = torch.device("cuda")
    # print(torch.cuda.is_available())
    torch.set_default_device("cuda")
    model = AudioModel(torch.nn.Tanh(), torch.nn.Mish(), torch.nn.ReLU())
    #original 2e-5
    #l am scazut dupa appx 26 de epoci
    #de la 2e-4
    learning_rate = 2e-5
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    t = 0
    try:
        checkpoint = torch.load(f'istorie antrenari/azi/model.model', weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        t = checkpoint['t']

        model.train()
        print('am incarcat model.model')
    except:
        print('nu am incarcat nici-un state dict')

    criterion = torch.nn.L1Loss()
    criterion.requires_grad_(True)
    mus = musdb.DB(subsets="train")

    #print(f'shape musdb {mus}')

    #original era Adam aici

    torch.set_grad_enabled(True)

    sdr = SignalDistortionRatio

    t0 = time.perf_counter()
    #print(mus[0].audio)

    debug = False
    while t < 1000:
        random.shuffle(mus.tracks)

        for song in tqdm(range(len(mus)), colour='#e0b0ff', file=sys.stdout, postfix= {'t': t} ):
            audio_original = mus[song].audio
            stems_original = mus[song].stems[(1, 2, 4, 3), :, :]

            # genereaza batches
            # eu sunt cel care antreneaza pe batch-uri si nu are leak-uri de memorie
            #x_batches = []
            y_true_batches = []
            total_batched = 0
            # pare sa mearga (trb sa incerc si cu batch_size-uri mai mari)
            while total_batched < audio_original.shape[0]:
                batch_size = random.randint(132300, 573300) # 3 - 13 secunde
                if audio_original.shape[0] - batch_size >= total_batched:
                    #x_batches.append(audio_original[total_batched: total_batched + batch_size, :])
                    y_true_batches.append(stems_original[:, total_batched: total_batched + batch_size, :])
                    total_batched += batch_size
                else:
                    #x_batches.append(audio_original[total_batched: , :])
                    y_true_batches.append(stems_original[:, total_batched:, :])
                    break

            #y_pred = None
            for y_batch in y_true_batches:

                #trb sa fac partea de separare benzi parte din model(ca sa nu mai rezolv probleme de memorie in loopul de training)
                aug = augment.Augment(torch.tensor(y_batch, device='cpu'))
                y_batch, x_true = aug()
                x_true = x_true.detach().numpy()

                x_true = torch.from_numpy(genereaza_tensor_din_stereo(x_true))
                x_true = x_true.to(torch.float32)
                y_batch = y_batch.to(torch.float32)



                x_true = x_true.to(device = "cuda")
                y_bar = model(x_true)
                y_batch = y_batch.to(device = 'cuda')

                loss = criterion(y_bar, y_batch)
                y_bar.detach()
                del y_bar
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()



                del y_batch
                del x_true

                gc.collect()

                if song % 10 == 9 or debug:
                    t1 = time.perf_counter()
                    #tqdm.write(f'dt = {t1 - t0}')
                    t0 = time.perf_counter()
                    tqdm.write(f't- {t}, song- {song}, mae: {loss.item()}')

                #daca nu fac evaluari nu am nevoide de y_bar
                '''
                y_bar = y_bar.to(device = 'cpu')
                if y_pred == None:
                    y_pred = y_bar
                else:
                    y_pred = torch.cat([y_pred, y_bar], dim = 1)

                del y_bar
                '''

            del y_true_batches

            torch.cuda.empty_cache()

            try:
                tqdm.write('scriu model.model nu ma inchide')
                torch.save({
                    't': t,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, f'istorie antrenari/azi/model.model')
                tqdm.write('gata')
            except:
                tqdm.write('nu am putut salva modelul')

        t += 1