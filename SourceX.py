import gc
import math
import os
import random
import sys

import musdb
import numpy as np
import torch
from scipy.signal import butter
from torchaudio.functional import filtfilt
from torchinfo import summary
from torchtune.modules import RotaryPositionalEmbeddings
from tqdm import tqdm

import augment

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


class ModifiedCelu(torch.nn.Module):
    def __init__(self, celu=None):
        super(ModifiedCelu, self).__init__()
        if celu is None:
            self.activation = torch.nn.CELU(alpha=1 / 8)
        else:
            self.activation = celu

    def forward(self, x):
        return self.activation(x + 1) - 1


# similar cu functia din demucs
def rescale_model(model, a):
    for sub in model.modules():
        if isinstance(sub, (torch.nn.Conv1d, torch.nn.ConvTranspose1d, torch.nn.Linear)):
            std = sub.weight.std().detach()
            scale = (std / a) ** 0.5
            sub.weight.data /= scale
            if sub.bias is not None:
                sub.bias.data /= scale


class EncoderModule(torch.nn.Module):
    def __init__(self, c_in, c_out, activation=None, stride=4):
        super().__init__()
        if activation is None:
            self.activation = torch.nn.CELU(alpha=1 / 8)
        else:
            self.activation = activation
        self.stride = stride

        self.conv1 = torch.nn.Conv1d(c_in, c_in, kernel_size=1)
        # self.glu = torch.nn.GLU(0)
        self.down = torch.nn.Conv1d(c_in, c_out, kernel_size=stride * 2, stride=stride)

        rescale_model(self, a=0.1)

    def pad_x(self, x: torch.Tensor, stride: int):
        if (x.shape[1]) % stride != 0:
            delta = stride - (x.shape[1] % stride)
            x = torch.cat([x, torch.zeros((x.shape[0], delta + (stride * 2)), device='cuda')], dim=1)
        else:
            x = torch.cat([x, torch.zeros((x.shape[0], stride * 2), device='cuda')], dim=1)
        return x

    def forward(self, x):
        x = self.activation(self.conv1(x))
        skip = x
        x = self.pad_x(x, stride=self.stride)
        x = self.activation(self.down(x))
        return x, skip


class DecoderModule(torch.nn.Module):
    def __init__(self, c_in, c_out, dim_skip=0, stride=4, activation=None, use_glu=True):
        super().__init__()
        if dim_skip == 0:
            self.dim_skip = c_in
        else:
            self.dim_skip = dim_skip

        self.use_glu = use_glu

        if activation is None:
            self.activation = torch.nn.CELU(1 / 8)
        else:
            self.activation = activation

        self.conv1 = torch.nn.Conv1d(c_in, c_in // 2, kernel_size=3)
        self.ups = torch.nn.ConvTranspose1d(c_in // 2, c_out, kernel_size=stride * 2, stride=stride)
        self.conv2 = torch.nn.Conv1d(c_out + dim_skip, c_out * 2, kernel_size=1)
        self.glu = torch.nn.GLU(0)

        rescale_model(self, a=0.1)

    def forward(self, x, skip):
        x = self.activation(self.conv1(x))
        x = self.activation(self.ups(x))
        x = torch.cat([x[:, :skip.shape[1]], skip], dim=0)
        x = self.conv2(x)
        if self.use_glu: x = self.glu(x)
        return x


class FFN(torch.nn.Module):
    def __init__(self, in_hidden, mid_hidden, output_hidden, activation=torch.nn.CELU(1 / 8)):
        super().__init__()
        self.dense1 = torch.nn.Linear(in_hidden, mid_hidden)
        self.activation = activation
        self.dense2 = torch.nn.Linear(mid_hidden, output_hidden)

    def forward(self, x):
        return self.dense2(self.activation(self.dense1(x)))


class AddNorm(torch.nn.Module):
    def __init__(self, norm_shape):
        super().__init__()
        self.ln = torch.nn.LayerNorm(norm_shape)

    def forward(self, x, y):
        return self.ln(y + x)


class TransformerEncoderLayer(torch.nn.Module):
    def __init__(self, hidden, fc_mid_hidden, heads, use_bias=False):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(hidden, heads, bias=use_bias)
        self.addnorm = AddNorm(hidden)
        self.ffn = FFN(hidden, fc_mid_hidden, hidden)

    def forward(self, x):
        y = self.addnorm(x, self.attention(x, x, x)[0])
        return self.addnorm(y, self.ffn(y))


class TransformerEncoder(torch.nn.Module):
    def __init__(self, hidden, ffn_mid_hidden, heads, use_bias=False):
        super().__init__()
        self.RoPE = RotaryPositionalEmbeddings(hidden // heads)
        self.hidden = hidden
        self.heads = heads
        self.layers = torch.nn.Sequential()
        self.l1 = TransformerEncoderLayer(hidden, ffn_mid_hidden, heads, use_bias=use_bias)
        self.l2 = TransformerEncoderLayer(hidden, ffn_mid_hidden, heads, use_bias=use_bias)
        self.l3 = TransformerEncoderLayer(hidden, ffn_mid_hidden, heads, use_bias=use_bias)
        # self.l4 = TransformerEncoderLayer(hidden, ffn_mid_hidden, heads, use_bias=use_bias)

    def forward(self, x):
        # print(x.shape)
        batch_size = 1  # Nu avem batch deci trb sa spunem ca avem 1
        time_length = x.shape[0]

        # Reshape from (time, hidden) to (batch, time, heads, head_dim)
        x = x.unsqueeze(0)  # Add batch dim -> (1, 217, 1000)
        x = x.view(batch_size, time_length, self.heads, self.hidden // self.heads)
        x = self.RoPE(x)
        x = x.view(batch_size, time_length, -1)  # (1, 217, 1000)
        x = x.squeeze(0)  # Back to (217, 1000) if needed

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        # x = self.l4(x)
        return x


# Aici se odihneste LSTM-ul, desi nu va mai fi parte din model vreodata,
# Acesta nu va pleca niciodata din LTM-ul nostru (Long term memry)
# Ca e gen "Long Short Term Memory"
# + 4/9/2025 - 5/26/2025 +
class BLSTMModule(torch.nn.Module):
    def __init__(self, nr_hidden):
        pass


class BandModel(torch.nn.Module):
    def __init__(self, filter_params=None):
        super().__init__()
        self.celu = torch.nn.CELU(1 / 8)
        self.modified_celu = ModifiedCelu(torch.nn.CELU(1 / 8))
        self.encoder_layer_1 = EncoderModule(2, 100, activation=ModifiedCelu(torch.nn.CELU(1 / 8)))
        self.encoder_layer_2 = EncoderModule(100, 300, activation=ModifiedCelu(torch.nn.CELU(1 / 8)))
        self.encoder_layer_3 = EncoderModule(300, 600, activation=ModifiedCelu(torch.nn.CELU(1 / 8)))
        self.encoder_layer_4 = EncoderModule(600, 750, activation=torch.nn.CELU(1 / 8))
        self.encoder_layer_5 = EncoderModule(750, 1000, activation=torch.nn.CELU(1 / 8))

        self.TransformerEncoder = TransformerEncoder(1000, 1400, 4)

        self.decoder_layer_5 = DecoderModule(1000, 750, activation=torch.nn.CELU(1 / 8), dim_skip=750)
        self.decoder_layer_4 = DecoderModule(750, 600, activation=torch.nn.CELU(1 / 8), dim_skip=600)
        self.decoder_layer_3 = DecoderModule(600, 300, activation=ModifiedCelu(torch.nn.CELU(1 / 8)), dim_skip=300)
        self.decoder_layer_2 = DecoderModule(300, 100, activation=ModifiedCelu(torch.nn.CELU(1 / 8)), dim_skip=100)
        self.decoder_layer_1 = DecoderModule(100, 8, dim_skip=2, activation=ModifiedCelu(torch.nn.CELU(1 / 8)),
                                             use_glu=False)

        # self.a = None
        # self.b = None
        self.use_filter = False
        if filter_params is not None:
            self.a = filter_params[1].to(device='cuda')
            self.a = self.a.to(dtype=torch.float32)
            self.b = filter_params[0].to(device='cuda')
            self.b = self.b.to(dtype=torch.float32)
            self.use_filter = True

    def forward(self, x: torch.Tensor):

        if self.use_filter:
            x = filtfilt(x, self.a, self.b)

        x, skip_1 = self.encoder_layer_1(x)
        x, skip_2 = self.encoder_layer_2(x)
        x, skip_3 = self.encoder_layer_3(x)
        x, skip_4 = self.encoder_layer_4(x)
        x, skip_5 = self.encoder_layer_5(x)

        x = torch.permute(x, (1, 0))

        x = self.TransformerEncoder(x)

        x = torch.permute(x, (1, 0))

        x = self.decoder_layer_5(x, skip_5)
        x = self.decoder_layer_4(x, skip_4)
        x = self.decoder_layer_3(x, skip_3)
        x = self.decoder_layer_2(x, skip_2)
        x = self.decoder_layer_1(x, skip_1)

        return x


class AudioModel(torch.nn.Module):
    def __init__(self, sample_rate=44100):
        super().__init__()

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
        # momentan x intra ca si t x c

        x = torch.permute(x, (1, 0))

        x_0 = self.band_0(x)
        x_1 = self.band_1(x)
        x_2 = self.band_2(x)
        x_3 = self.band_3(x)

        x = torch.cat([x_0, x_1, x_2, x_3], dim=0)

        x = torch.permute(x, (1, 0))
        x = self.FC(x)
        x = torch.permute(x, (1, 0))
        return x


# de pe forum pytorch pentru ca nu exista functie implementata
# pentru a muta optimizatorul pe GPU deci doar se pune functia
# asta
def optimizer_to(optim, device):
    for param in optim.state.values():
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
    learning_rate = 1e-5
    model = AudioModel()
    print(summary(model))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    torch.set_default_device("cuda")
    t = 0
    try:
        checkpoint = torch.load(f'istorie antrenari/azi/model.model', weights_only=True, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        t = checkpoint['t']

        torch.cuda.empty_cache()
        gc.collect()

        print('am incarcat model.model')
    except:
        print('eroare la incarcare model.model')

    model.to(device='cuda')
    optimizer_to(optimizer, 'cuda')

    criterion = torch.nn.L1Loss()
    criterion.requires_grad_(True)
    mus = musdb.DB(subsets="train")

    aug = augment.Augment()

    batch_size = 44100 * 5  # 5 secunde

    debug = False
    while t < 1000:
        random.shuffle(mus.tracks)
        postfix = {
            't': t,
        }

        songs_in_a_batch = 5
        # progress_bar = tqdm(range(len(mus)), colour='#e0b0ff', file=sys.stdout, postfix=postfix)
        total_loss_epoch = 0
        nr_batches_epoch = 0

        with tqdm(total=math.ceil(len(mus) / songs_in_a_batch), colour='#e0b0ff', file=sys.stdout,
                  postfix=postfix) as pbar:
            for song_idx in range(len(mus)):
                if song_idx % songs_in_a_batch != 0:
                    continue

                stems_original = mus[song_idx].stems[(1, 2, 4, 3), :, :]
                try:
                    for i in range(1, songs_in_a_batch):
                        stems_original = np.concatenate((stems_original, mus[song_idx + i].stems[(1, 2, 4, 3), :, :]),
                                                        axis=1)
                except:
                    pass

                s1_batches = []
                s2_batches = []
                s3_batches = []
                s4_batches = []
                total_batched = 0
                while total_batched < stems_original.shape[1]:
                    if stems_original.shape[1] - batch_size >= total_batched:
                        s1_batches.append(
                            torch.from_numpy(stems_original[0, total_batched: total_batched + batch_size, :]))
                        s2_batches.append(
                            torch.from_numpy(stems_original[1, total_batched: total_batched + batch_size, :]))
                        s3_batches.append(
                            torch.from_numpy(stems_original[2, total_batched: total_batched + batch_size, :]))
                        s4_batches.append(
                            torch.from_numpy(stems_original[3, total_batched: total_batched + batch_size, :]))
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
                    # t0 = time.perf_counter()
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

                    # adauga detach aici
                    loss = criterion(y_bar, y_batch)
                    total_loss += loss.detach().item()
                    total_loss_epoch += loss.detach().item()

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    # print(time.perf_counter() - t0)

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
                pbar.set_description(
                    f'avg loss for last batch: {total_loss / nr_batches}, for epoch:{total_loss_epoch / nr_batches_epoch}')
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
