import gc
import math
import os
import random
import sys
import time
from symtable import Class

import musdb
import numpy as np
import scipy.signal
import torch
from scipy.signal import butter
from torch import newaxis
from torchaudio.functional import filtfilt
from tqdm import tqdm

import augment

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


#aproape ca in Demucs
def rescale_model(model, a):
    for sub in model.modules():
        if isinstance(sub, (torch.nn.Conv1d, torch.nn.ConvTranspose1d, torch.nn.Linear)):
            std = sub.weight.std().detach()
            scale = (std/ a) ** 0.5
            sub.weight.data /= scale
            if sub.bias is not None:
                sub.bias.data /= scale


# FA FARA LAZY PUTUROSULE
class EncoderModule(torch.nn.Module):
    def __init__(self,c_in, c_out, skip_first_mish=False):
        super().__init__()
        self.skip_first_mish = skip_first_mish
        self.conv1 = torch.nn.Conv1d(c_in, c_in * 2, kernel_size=1)
        self.down = torch.nn.Conv1d(c_in * 2, c_out, kernel_size=8, stride=4)
        rescale_model(self, a=0.1)


    def pad_x(self, x: torch.Tensor, stride: int):
        #rescrie asta sa mearga cu conv1d
        if (x.shape[1]) % stride != 0:
            delta = stride - (x.shape[1] % stride)
            x = torch.cat([x, torch.zeros((x.shape[0], delta + (stride * 2)), device='cuda')], dim=1)
        else:
            x = torch.cat([x, torch.zeros((x.shape[0], stride * 2), device='cuda')], dim=1)
        return x

    def forward(self, x):
        x = self.conv1(x)
        if self.skip_first_mish is False:
            x = torch.nn.functional.mish(x)
        skip = x
        x = self.pad_x(x, stride=4)
        x = torch.nn.functional.mish(self.down(x))
        return x, skip

class DecoderModule(torch.nn.Module):
    def __init__(self, c_in, c_out, dim_skip = None):
        super().__init__()
        if dim_skip is None:
            self.conv1 = torch.nn.Conv1d(c_in, c_in * 2, kernel_size=1)
            self.ups = torch.nn.ConvTranspose1d(c_in * 2, c_out, kernel_size=4, stride=4)
            self.conv2 = torch.nn.Sequential (
                torch.nn.Conv1d(c_out * 3, c_out * 2, kernel_size=1),
                torch.nn.GLU(0),
            )
        else:
            self.conv1 = torch.nn.Conv1d(c_in, c_in * 2, kernel_size=1)
            self.ups = torch.nn.ConvTranspose1d(c_in * 2, c_out, kernel_size=4, stride=4)
            self.conv2 = torch.nn.Sequential(
                torch.nn.Conv1d(c_out + dim_skip, c_out * 2, kernel_size=1),
                torch.nn.GLU(0),
            )
        rescale_model(self, a=0.1)

    def forward(self, x, skip):
        x = torch.nn.functional.mish(self.conv1(x))
        x = torch.nn.functional.mish(self.ups(x))
        x = torch.cat([x[:, :skip.shape[1]], skip], dim=0)
        x = self.conv2(x)
        return x

class LastDecoderModule(torch.nn.Module):
    def __init__(self, c_in, c_out, skip_dim = 4):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(c_in, c_in * 2, kernel_size=1)
        self.ups = torch.nn.ConvTranspose1d(c_in * 2, c_out, kernel_size=4, stride=4)
        self.conv2 = torch.nn.Sequential (
            torch.nn.Conv1d(c_out + skip_dim, c_out, kernel_size=1),
        )
        rescale_model(self, a=0.1)

    def forward(self, x, skip):
        x = torch.nn.functional.mish(self.conv1(x))
        x = torch.nn.functional.mish(self.ups(x))
        x = torch.cat([x[:, :skip.shape[1]], skip], dim=0)
        x = self.conv2(x)
        return x

class BLSTMModule(torch.nn.Module):
    def __init__(self, nr_hidden):
        super().__init__()
        self.FC1 = torch.nn.Sequential(
            torch.nn.Linear(nr_hidden, nr_hidden), torch.nn.Mish(),
            torch.nn.Linear(nr_hidden, nr_hidden)
        )
        self.att1 = torch.nn.MultiheadAttention(nr_hidden, 4)
        self.BLSTM = torch.nn.LSTM(input_size= nr_hidden, hidden_size=nr_hidden,bidirectional=True, num_layers=2)
        self.FC2 = torch.nn.Sequential(
            torch.nn.Linear(nr_hidden * 2, nr_hidden),
        )
        self.att2 = torch.nn.MultiheadAttention(nr_hidden, 4)
        self.FC3 = torch.nn.Sequential(
            torch.nn.Linear(nr_hidden, nr_hidden), torch.nn.Mish(),
            torch.nn.Linear(nr_hidden, nr_hidden), torch.nn.Mish(),
        )
        rescale_model(self, a=0.1)

    def forward(self, x):
        x = self.FC1(x)
        x = self.att1(x, x, x)[0]
        x = self.BLSTM(x)[0]
        x = self.FC2(x)
        x = self.att2(x, x, x)[0]
        x = self.FC3(x)
        return x

class BandModel(torch.nn.Module):
    def __init__(self, filter_params=None):
        super().__init__()
        self.encoder_layer_1 = EncoderModule(2, 8, skip_first_mish=True)
        self.encoder_layer_2 = EncoderModule(8, 32)
        self.encoder_layer_3 = EncoderModule(32, 128)
        self.encoder_layer_4 = EncoderModule(128, 512)
        self.encoder_layer_5 = EncoderModule(512, 1024)

        self.BLSTM = BLSTMModule(1024)

        self.decoder_layer_5 = DecoderModule(1024, 512)
        self.decoder_layer_4 = DecoderModule(512, 128)
        self.decoder_layer_3 = DecoderModule(128, 32)
        self.decoder_layer_2 = DecoderModule(32, 16, dim_skip = 16)
        self.decoder_layer_1 = LastDecoderModule(16, 16, skip_dim=4)

        self.a = None
        self.b = None
        if filter_params is not None:
            self.a = filter_params[1].to(device='cuda')
            self.a = self.a.to(dtype=torch.float32)
            self.b = filter_params[0].to(device='cuda')
            self.b = self.b.to(dtype=torch.float32)



    def forward(self, x: torch.Tensor):

        if self.a is not None and self.b is not None:
            x = filtfilt(x, self.a, self.b)

        x, skip_1 = self.encoder_layer_1(x)
        x, skip_2 = self.encoder_layer_2(x)
        x, skip_3 = self.encoder_layer_3(x)
        x, skip_4 = self.encoder_layer_4(x)
        x, skip_5 = self.encoder_layer_5(x)

        x = torch.permute(x, (1, 0))

        x = self.BLSTM(x)

        x = torch.permute(x, (1, 0))

        x = self.decoder_layer_5(x, skip_5)
        x = self.decoder_layer_4(x, skip_4)
        x = self.decoder_layer_3(x, skip_3)
        x = self.decoder_layer_2(x, skip_2)
        x = self.decoder_layer_1(x, skip_1)

        return x

class AudioModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        sample_rate = 44100


        self.band_0 = BandModel()

        b, a = butter(3, 400 / sample_rate * 2, 'low', analog=False)
        self.band_1 = BandModel((torch.from_numpy(b), torch.from_numpy(a)))

        b, a = butter(3, np.array([400, 1900]) / sample_rate * 2, 'band', analog=False)
        self.band_2 = BandModel((torch.from_numpy(b), torch.from_numpy(a)))

        b, a = butter(3, 1900 / sample_rate * 2, 'high', analog=False)
        self.band_3 = BandModel((torch.from_numpy(b), torch.from_numpy(a)))

        self.FC = torch.nn.Sequential(
            torch.nn.Linear(16 * 4, 16 * 4), torch.nn.Tanh(),
            torch.nn.Linear(16 * 4, 16 * 3), torch.nn.Tanh(),
            torch.nn.Linear(16 * 3, 16 * 2), torch.nn.Tanh(),
            torch.nn.Linear(16 * 2, 16),
            torch.nn.Linear(16, 8)
        )

        rescale_model(self, 0.1)

    def forward(self, x):

        #momentan x intra ca si t x c


        x = torch.permute(x, (1, 0))

        x_0 = self.band_0(x)
        x_1 = self.band_1(x)
        x_2 = self.band_2(x)
        x_3 = self.band_3(x)

        x = torch.cat([x_0, x_1, x_2, x_3], dim = 0)

        x = torch.permute(x, (1, 0))
        x = self.FC(x)
        x = torch.permute(x, (1, 0))
        return x

# cod de pe https://github.com/pytorch/pytorch/issues/7415#issuecomment-693424574
def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


if __name__ == '__main__':

    dtype = torch.float32
    device = torch.device("cuda")
    learning_rate = 2e-4
    model = AudioModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    torch.set_default_device("cuda")
    nr_param = sum(parameter.numel() for parameter in model.parameters())
    print(nr_param)
    # original 2e-5

    t = 0
    try:
        checkpoint = torch.load(f'istorie antrenari/azi/model.model', weights_only=True, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        t = checkpoint['t']

        torch.cuda.empty_cache()
        gc.collect()


        #model.train()
        print('am incarcat model.model')
    except:
        print('nu am incarcat nici-un state dict (nu uita sa stergi ce model.model deja exista (daca exista) ca s-ar putea sa creeze probleme :3)')

    model.to(device='cuda')
    optimizer_to(optimizer, 'cuda')

    criterion = torch.nn.L1Loss()
    criterion.requires_grad_(True)
    mus = musdb.DB(subsets="train")
    #torch.set_grad_enabled(True)

    aug = augment.Augment()

    batch_size = 44100 * 5 # 6 secunde

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
                s1_batches = []
                s2_batches = []
                s3_batches = []
                s4_batches = []
                total_batched = 0
                while total_batched < stems_original.shape[1]:
                    if stems_original.shape[1] - batch_size >= total_batched:
                        s1_batches.append(torch.from_numpy(stems_original[0, total_batched: total_batched + batch_size, :]))
                        s2_batches.append(torch.from_numpy(stems_original[1, total_batched: total_batched + batch_size, :]))
                        s3_batches.append(torch.from_numpy(stems_original[2, total_batched: total_batched + batch_size, :]))
                        s4_batches.append(torch.from_numpy(stems_original[3, total_batched: total_batched + batch_size, :]))
                        total_batched += batch_size
                    else:
                        break

                random.shuffle(s1_batches)
                random.shuffle(s2_batches)
                random.shuffle(s3_batches)
                random.shuffle(s4_batches)

                total_loss = 0
                nr_batches = len(s1_batches)
                nr_batched_batches = 2
                nr_batches_epoch += nr_batches
                for idx in range(len(s1_batches)):
                    s1 = s1_batches.pop(0)
                    s2 = s2_batches.pop(0)
                    s3 = s3_batches.pop(0)
                    s4 = s4_batches.pop(0)
                    y_batch = torch.stack([s1, s2, s3, s4], dim=0)
                    y_batch = y_batch.to(device='cuda')
                    y_batch, x_true, err = aug.forward(y_batch)
                    if err == 1:
                        continue
                    x_true = x_true.to(dtype=torch.float32)
                    y_batch = y_batch.to(dtype=torch.float32)




                    y_bar = model(x_true)
                    tensor_nou = torch.zeros(8, y_batch.shape[1])
                    for i in range(4):
                        tensor_nou[i * 2] = y_batch[i, :, 0]
                        tensor_nou[(i * 2) + 1] = y_batch[i, :, 1]

                    y_batch = tensor_nou
                    del tensor_nou


                    loss = criterion(y_bar, y_batch)
                    total_loss += loss.item()
                    total_loss_epoch += loss.item()

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    del y_batch
                    del s1
                    del s2
                    del s3
                    del s4


                del s1_batches
                del s2_batches
                del s3_batches
                del s4_batches

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
        try:
            torch.save({
                't': t,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'istorie antrenari/azi/model.model')
        except:
            tqdm.write('nu am putut salva modelul')
