import gc
import math
import os
import random
import sys
import time

import musdb
import numpy as np
import torch
from scipy.signal import butter, filtfilt
from torch import newaxis
from tqdm import tqdm

import augment

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


def genereaza_strat_banda(tensor, filter):
    y = filtfilt(filter[0], filter[1], tensor, 0)
    return y


def genereaza_tensor_din_stereo(tensor, fs=44100):

    # (ex) 2.1.1 Separarea pe benzi a semnalului original

    output_3_axe_nou = tensor[..., newaxis]
    output = tensor
    nyq = 0.5 * fs

    # primele 3 benzi o sa fie low - mid - high
    cutoff = 400

    normal_cutoff = cutoff / nyq
    b, a = butter(5, normal_cutoff, btype='low', analog=False)
    output_3_axe_nou = np.concatenate([output_3_axe_nou, genereaza_strat_banda(output, (b, a))[..., newaxis]], axis=2)

    cutoff = np.array([400, 1900])
    normal_cutoff = cutoff / nyq
    b, a = butter(5, normal_cutoff, btype='band', analog=False)
    output_3_axe_nou = np.concatenate([output_3_axe_nou, genereaza_strat_banda(output, (b, a))[..., newaxis]], axis=2)

    cutoff = 1900
    normal_cutoff = cutoff / nyq
    b, a = butter(5, normal_cutoff, btype='high', analog=False)
    output_3_axe_nou = np.concatenate([output_3_axe_nou, genereaza_strat_banda(output, (b, a))[..., newaxis]], axis=2)

    # si dupa incepem cu benzile "harmonice"

    cutoff = 640
    normal_cutoff = cutoff / nyq
    b, a = butter(5, normal_cutoff, btype='low', analog=False)
    output_3_axe_nou = np.concatenate([output_3_axe_nou, genereaza_strat_banda(output, (b, a))[..., newaxis]], axis=2)

    cutoff = np.array([640, 1280])
    normal_cutoff = cutoff / nyq
    b, a = butter(5, normal_cutoff, btype='band', analog=False)
    output_3_axe_nou = np.concatenate([output_3_axe_nou, genereaza_strat_banda(output, (b, a))[..., newaxis]], axis=2)

    cutoff = np.array([1280, 2560])
    normal_cutoff = cutoff / nyq
    b, a = butter(5, normal_cutoff, btype='band', analog=False)
    output_3_axe_nou = np.concatenate([output_3_axe_nou, genereaza_strat_banda(output, (b, a))[..., newaxis]], axis=2)

    cutoff = np.array([2560, 5120])
    normal_cutoff = cutoff / nyq
    b, a = butter(5, normal_cutoff, btype='band', analog=False)
    output_3_axe_nou = np.concatenate([output_3_axe_nou, genereaza_strat_banda(output, (b, a))[..., newaxis]], axis=2)

    cutoff = np.array([5120, 10240])
    normal_cutoff = cutoff / nyq
    b, a = butter(5, normal_cutoff, btype='band', analog=False)
    output_3_axe_nou = np.concatenate([output_3_axe_nou, genereaza_strat_banda(output, (b, a))[..., newaxis]], axis=2)

    cutoff = 10240
    normal_cutoff = cutoff / nyq
    b, a = butter(5, normal_cutoff, btype='high', analog=False)
    output_3_axe_nou = np.concatenate([output_3_axe_nou, genereaza_strat_banda(output, (b, a))[..., newaxis]], axis=2)

    return output_3_axe_nou

#aproape ca in Demucs
def rescale_model(model, a):
    for sub in model.modules():
        if isinstance(sub, (torch.nn.Conv2d, torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
            std = sub.weight.std().detach()
            scale = (std/ a) ** 0.5
            sub.weight.data /= scale
            if sub.bias is not None:
                sub.bias.data /= scale

class AudioModel(torch.nn.Module):
    def __init__(self, mish_like):
        super().__init__()

        # --- encoders:
        # layer 1
        self.enc_f_1 = torch.nn.Sequential(torch.nn.Conv2d(10, 60, (1, 2), padding='same', padding_mode='circular'),
                                           mish_like)
        self.enc_dwn_1 = torch.nn.Sequential(torch.nn.Conv1d(60, 120, 8, stride=4), mish_like)

        # layer 2
        self.enc_f_2 = torch.nn.Sequential(torch.nn.Conv2d(120, 120, (1, 2), padding='same', padding_mode='circular'),
                                           mish_like)
        self.enc_dwn_2 = torch.nn.Sequential(torch.nn.Conv1d(120, 200, 8, stride=4), mish_like)

        # layer 3
        self.enc_f_3 = torch.nn.Sequential(torch.nn.Conv2d(200, 200, (1, 2), padding='same', padding_mode='circular'),
                                           mish_like)
        self.enc_dwn_3 = torch.nn.Sequential(torch.nn.Conv1d(200, 600, 8, stride=4), mish_like)

        # layer 4
        self.enc_f_4 = torch.nn.Sequential(torch.nn.Conv2d(600, 600, (3, 2), padding='same', padding_mode='circular'),
                                           mish_like)
        self.enc_dwn_4 = torch.nn.Sequential(torch.nn.Conv1d(600, 1200, 8, stride=4), mish_like)

        # layer 5
        self.enc_f_5 = torch.nn.Sequential(torch.nn.Conv2d(1200, 1200, (3, 2), padding='same', padding_mode='circular'),
                                           mish_like)
        self.enc_dwn_5 = torch.nn.Sequential(torch.nn.Conv1d(1200, 1800, 8, stride=4), mish_like)

        self.blstm = torch.nn.LSTM(1800, 1800, bidirectional=True, num_layers=2)
        self.blstm_linear = torch.nn.Sequential(torch.nn.Linear(3600, 3000), mish_like,
                                                torch.nn.Linear(3000, 2500), mish_like,
                                                torch.nn.Linear(2500, 2000), mish_like,
                                                torch.nn.Linear(2000, 1200), mish_like)

        # --- decoders:
        # layer 5
        self.dec_f1_5 = torch.nn.Sequential(torch.nn.Conv2d(1200, 1200, (1, 2), padding='same', padding_mode='circular'),
                                            mish_like)
        self.dec_ups_5 = torch.nn.Sequential(torch.nn.ConvTranspose1d(1200, 1200, 4, 4), mish_like)
        self.dec_f2_5 = torch.nn.Sequential(torch.nn.Conv2d(2400, 1200, (3, 2), padding='same', padding_mode='circular'),
                                            torch.nn.GLU(0))

        # layer 4
        self.dec_f1_4 = torch.nn.Sequential(torch.nn.Conv2d(600, 600, (1, 2), padding='same', padding_mode='circular'),
                                            mish_like)
        self.dec_ups_4 = torch.nn.Sequential(torch.nn.ConvTranspose1d(600, 600, 4, 4), mish_like)
        self.dec_f2_4 = torch.nn.Sequential(torch.nn.Conv2d(1200, 400, (3, 2), padding='same', padding_mode='circular'),
                                            torch.nn.GLU(0))

          # layer 3
        self.dec_f1_3 = torch.nn.Sequential(torch.nn.Conv2d(200, 200, (1, 2), padding='same', padding_mode='circular'),
                                            mish_like)
        self.dec_ups_3 = torch.nn.Sequential(torch.nn.ConvTranspose1d(200, 200, 4, 4), mish_like)
        self.dec_f2_3 = torch.nn.Sequential(torch.nn.Conv2d(400, 240, (1, 2), padding='same', padding_mode='circular'),
                                            torch.nn.GLU(0))

        # layer 2
        self.dec_f1_2 = torch.nn.Sequential(torch.nn.Conv2d(120, 120, (1, 2), padding='same', padding_mode='circular'),
                                            mish_like)
        self.dec_ups_2 = torch.nn.Sequential(torch.nn.ConvTranspose1d(120, 120, 4, 4), mish_like)
        self.dec_f2_2 = torch.nn.Sequential(torch.nn.Conv2d(240, 120, (1, 2), padding='same', padding_mode='circular'),
                                            torch.nn.GLU(0))

        # layer 1
        self.dec_f1_1 = torch.nn.Sequential(torch.nn.Conv2d(60, 60, (1, 2), padding='same', padding_mode='circular'), mish_like)
        self.dec_ups_1 = torch.nn.Sequential(torch.nn.ConvTranspose1d(60, 60, 4, 4), mish_like)
        self.dec_f2_1 = torch.nn.Sequential(torch.nn.Conv2d(120, 4, (1, 2), padding='same', padding_mode='circular'))

        rescale_model(self, a=0.1)

    def pad_x(self, x: torch.Tensor, stride: int):
        if (x.shape[2]) % stride != 0:
            delta = stride - (x.shape[2] % stride)
            x = torch.cat([x, torch.zeros((x.shape[0], x.shape[1], delta + (stride * 2)), device='cuda')], dim=2)
        else:
            x = torch.cat([x, torch.zeros((x.shape[0], x.shape[1], stride * 2), device='cuda')], dim=2)
        return x

    def forward(self, x: torch.Tensor):
        skip = []
        # x intra ca (T, C, B)

        # enc layer 1: -------------------------------------------------------------------
        # pentru Conv2d vrem(B, T, C) (Permutatie (T, C, B) -> (B, T, C))
        x = x.permute(2, 0, 1)
        x = self.enc_f_1(x)

        # pentru Conv1d vrem(C, B, T) (Permutatie (B, T, C) -> (C, B, T))
        x = x.permute(2, 0, 1)
        skip.append(x)
        x = self.pad_x(x, 4)
        x = self.enc_dwn_1(x)

        # pentru urmatorul Conv2d vrem (B, T, C) (Permutatie (C, B, T) -> (B, T, C)))
        x = x.permute(1, 2, 0)

        # enc layer 2: -------------------------------------------------------------------
        x = self.enc_f_2(x)

        # pentru Conv1d vrem(C, B, T) (Permutatie (B, T, C) -> (C, B, T))
        x = x.permute(2, 0, 1)
        skip.append(x)
        x = self.pad_x(x, 4)
        x = self.enc_dwn_2(x)

        # pentru urmatorul Conv2d vrem (B, T, C) (Permutatie (C, B, T) -> (B, T, C)))
        x = x.permute(1, 2, 0)

        # enc layer 3: -------------------------------------------------------------------
        x = self.enc_f_3(x)

        # pentru Conv1d vrem(C, B, T) (Permutatie (B, T, C) -> (C, B, T))
        x = x.permute(2, 0, 1)
        skip.append(x)
        x = self.pad_x(x, 4)
        x = self.enc_dwn_3(x)

        # pentru urmatorul Conv2d vrem (B, T, C) (Permutatie (C, B, T) -> (B, T, C)))
        x = x.permute(1, 2, 0)

        # enc layer 4: -------------------------------------------------------------------
        x = self.enc_f_4(x)

        # pentru Conv1d vrem(C, B, T) (Permutatie (B, T, C) -> (C, B, T))
        x = x.permute(2, 0, 1)
        skip.append(x)
        x = self.pad_x(x, 4)
        x = self.enc_dwn_4(x)

        # pentru urmatorul Conv2d vrem (B, T, C) (Permutatie (C, B, T) -> (B, T, C)))
        x = x.permute(1, 2, 0)

        # enc layer 5: -------------------------------------------------------------------
        x = self.enc_f_5(x)

        # pentru Conv1d vrem(C, B, T) (Permutatie (B, T, C) -> (C, B, T))
        x = x.permute(2, 0, 1)
        skip.append(x)
        x = self.pad_x(x, 4)
        x = self.enc_dwn_5(x)

        # blstm -----------------------------------------------------------------------
        # pentru blstm vrem (T, C, B) (Permutatie (C, B, T) -> (T, C, B))
        x = x.permute(2, 0, 1)

        x = self.blstm(x)[0]

        x = self.blstm_linear(x)

        # pentru urmatorul filtru vrem (B, T, C) (Permutatie (T, C, B) -> (B, T, C))
        x = x.permute(2, 0, 1)

        # dec layer 5: -----------------------------------------------------------------
        x = self.dec_f1_5(x)
        # pentru Conv1d vrem(C, B, T) (Permutatie (B, T, C) -> (C, B, T))
        x = x.permute(2, 0, 1)
        x = self.dec_ups_5(x)
        skip_tensor = skip.pop(-1)
        x = torch.cat([x[:, :, :skip_tensor.shape[2]], skip_tensor], dim=1)
        x = x.permute(1, 2, 0)

        x = self.dec_f2_5(x)

        # dec layer 4: -----------------------------------------------------------------
        x = self.dec_f1_4(x)
        # pentru Conv1d vrem(C, B, T) (Permutatie (B, T, C) -> (C, B, T))
        x = x.permute(2, 0, 1)

        x = self.dec_ups_4(x)
        skip_tensor = skip.pop(-1)
        x = torch.cat([x[:, :, :skip_tensor.shape[2]], skip_tensor], dim=1)
        # pentru Conv2d vrem (B, T, C) (Permutatie (C, B, T) -> (B, T, C)))
        x = x.permute(1, 2, 0)

        x = self.dec_f2_4(x)

        # dec layer 3: -----------------------------------------------------------------
        x = self.dec_f1_3(x)
        # pentru Conv1d vrem(C, B, T) (Permutatie (B, T, C) -> (C, B, T))
        x = x.permute(2, 0, 1)

        x = self.dec_ups_3(x)
        skip_tensor = skip.pop(-1)
        x = torch.cat([x[:, :, :skip_tensor.shape[2]], skip_tensor], dim=1)
        # pentru Conv2d vrem (B, T, C) (Permutatie (C, B, T) -> (B, T, C)))
        x = x.permute(1, 2, 0)

        x = self.dec_f2_3(x)

        # dec layer 2: -----------------------------------------------------------------
        x = self.dec_f1_2(x)
        # pentru Conv1d vrem(C, B, T) (Permutatie (B, T, C) -> (C, B, T))
        x = x.permute(2, 0, 1)

        x = self.dec_ups_2(x)
        skip_tensor = skip.pop(-1)
        x = torch.cat([x[:, :, :skip_tensor.shape[2]], skip_tensor], dim=1)
        # pentru Conv2d vrem (B, T, C) (Permutatie (C, B, T) -> (B, T, C)))
        x = x.permute(1, 2, 0)
        x = self.dec_f2_2(x)

        # dec layer 1: -----------------------------------------------------------------
        x = self.dec_f1_1(x)
        # pentru Conv1d vrem(C, B, T) (Permutatie (B, T, C) -> (C, B, T))
        x = x.permute(2, 0, 1)

        x = self.dec_ups_1(x)
        skip_tensor = skip.pop(-1)
        x = torch.cat([x[:, :, :skip_tensor.shape[2]], skip_tensor], dim=1)

        # pentru Conv2d vrem (B, T, C) (Permutatie (C, B, T) -> (B, T, C)))
        x = x.permute(1, 2, 0)

        x = self.dec_f2_1(x)

        #del skip_tensor
        return x


if __name__ == '__main__':

    dtype = torch.float32
    device = torch.device("cuda")
    torch.set_default_device("cuda")
    model = AudioModel(torch.nn.Mish())
    # original 2e-5
    learning_rate = 2e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    t = 0
    try:
        checkpoint = torch.load(f'istorie antrenari/azi/model.model', weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        t = checkpoint['t']

        #model.train()
        print('am incarcat model.model')
    except:
        print('nu am incarcat nici-un state dict (nu uita sa stergi ce model.model deja exista (daca exista) ca s-ar putea sa creeze probleme :3)')

    criterion = torch.nn.L1Loss()
    criterion.requires_grad_(True)
    mus = musdb.DB(subsets="train")
    #torch.set_grad_enabled(True)

    aug = augment.Augment()


    debug = False
    while t < 1000:
        random.shuffle(mus.tracks)
        postfix = {
            't' : t,
        }

        songs_in_a_batch = 5
        #progress_bar = tqdm(range(len(mus)), colour='#e0b0ff', file=sys.stdout, postfix=postfix)
        total_loss_epoch = 0
        nr_batches_epoch = 0

        with tqdm(total = math.ceil(len(mus) / songs_in_a_batch), colour='#e0b0ff', file=sys.stdout, postfix=postfix) as pbar:
            for song_idx in range(len(mus)):
                #cand am incercat sa citesc random chunk-uri din toate melodiile
                #statea 34 de ore la o epoca T_T
                #pbar.write(f'{song_idx}')
                if song_idx % songs_in_a_batch != 0:
                    continue
                #print(song_idx)

                stems_original = mus[song_idx].stems[(1, 2, 4, 3), :, :]
                try:
                    for i in range(1, songs_in_a_batch):
                        stems_original = np.concatenate((stems_original, mus[song_idx + i].stems[(1, 2, 4, 3), :, :]), axis=1)
                except:
                    pass


                #daca intentionez sa tin batch-size-ul constant atunci
                #ar trebui sa ma scap de lista y_true_batches si sa folosesc
                #un ndarray
                y_true_batches = []
                total_batched = 0
                while total_batched < stems_original.shape[1]:
                    #batch_size = random.randint(220500, 264600)  # 5 - 6 secunde
                    batch_size = 176400 # 4 secunde
                    if stems_original.shape[1] - batch_size >= total_batched:
                        y_true_batches.append(stems_original[:, total_batched: total_batched + batch_size, :])
                        total_batched += batch_size
                    else:
                        y_true_batches.append(stems_original[:, total_batched:, :])
                        break

                random.shuffle(y_true_batches)

                total_loss = 0
                nr_batches = len(y_true_batches)
                nr_batches_epoch += nr_batches
                #as putea face cu pop in loc de for pt a economisii memorie
                for y_batch in y_true_batches:
                    #t0 = time.perf_counter()
                    y_batch, x_true, err = aug.forward(y_batch)
                    if err == 1:
                        continue
                    #if np.any(np.isnan(x_true)) == 1:
                    #    tqdm.write('Warning, aug a returnat np.array cu Nan')
                    #    tqdm.write(f'{x_true}')
                    #    continue
                    x_true = genereaza_tensor_din_stereo(x_true)
                    x_true = torch.tensor(x_true)
                    x_true = x_true.to(torch.float32)
                    y_batch = torch.tensor(y_batch)
                    y_batch = y_batch.to(torch.float32)

                    x_true = x_true.to(device="cuda")
                    y_bar = model(x_true)
                    y_batch = y_batch.to(device='cuda')

                    loss = criterion(y_bar, y_batch)
                    total_loss += loss.item()
                    total_loss_epoch += loss.item()

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()


                    #tqdm.write(f'time: {time.perf_counter() - t0}')

                del y_true_batches
                #torch.cuda.empty_cache()

                try:
                    torch.save({
                        't': t,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, f'istorie antrenari/azi/model.model')
                except:
                    tqdm.write('nu am putut salva modelul')
                pbar.set_description(f'avg loss for last batch: {total_loss / nr_batches}, for epoch:{total_loss_epoch / nr_batches_epoch}')
                pbar.update(1)

                gc.collect()
        t += 1
