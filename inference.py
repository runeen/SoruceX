import gc
import io
import zipfile

import numpy
import torch
from scipy.io.wavfile import read, write

import SourceX

dtype = torch.float32
device = torch.device("cuda")
# print(torch.cuda.is_available())
model = SourceX.AudioModel()
try:
    checkpoint = torch.load(f'istorie antrenari/azi/model.model', weights_only=True, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    t = checkpoint['t']

    torch.cuda.empty_cache()
    gc.collect()

    model.eval()
except:
    print('nu am incarcat nici-un state dict')

torch.set_default_device("cuda")
model.to(device='cuda')

def separate_into_dict(song, rate=44100):
    with torch.no_grad():

        input_file = song / numpy.iinfo(numpy.int16).max
        #print(input_file)

        # genereaza batches
        # posibil memory leak
        x_batches = []
        total_batched = 0
        batch_size = 2646000  # 1 min
        while total_batched < input_file.shape[0]:
            if input_file.shape[0] - batch_size >= total_batched:
                x_batches.append(input_file[total_batched: total_batched + batch_size, :])
                total_batched += batch_size
            else:
                x_batches.append(input_file[total_batched:, :])
                break

        y_pred = None
        for x_batch in x_batches:
            #print('started batch')
            x_batch = torch.tensor(x_batch)
            x_batch = x_batch.to(torch.float32)
            x_batch = x_batch.to(device="cuda")
            output = model(x_batch)

            # s a intamplat ceva funky si acum sunt speriat
            del x_batch
            output = output.to(device='cpu')
            if y_pred == None:
                y_pred = output
            else:
                y_pred = torch.cat([y_pred, output], dim=1)

            del output

        y_pred_np = y_pred.detach().numpy()
        print(y_pred_np)

        #write(f'output/drums.wav', rate, (y_pred_np[(0, 1), :] * 32767).T.astype(numpy.int16))
        #write(f'output/bass.wav', rate, (y_pred_np[(2, 3), :] * 32767).T.astype(numpy.int16))
        #write(f'output/vocals.wav', rate, (y_pred_np[(4, 5), :] * 32767).T.astype(numpy.int16))
        #write(f'output/other.wav', rate, (y_pred_np[(6, 7), :] * 32767).T.astype(numpy.int16))

        output = {
            'rate': rate,
            'drums': (y_pred_np[(0, 1), :] * 32767).T.astype(numpy.int16),
            'bass': (y_pred_np[(2, 3), :] * 32767).T.astype(numpy.int16),
            'vocals': (y_pred_np[(4, 5), :] * 32767).T.astype(numpy.int16),
            'other': (y_pred_np[(6, 7), :] * 32767).T.astype(numpy.int16)
        }

        del y_pred

        gc.collect()

        return output

def write_dict_to_files(separated_stems):
    rate = separated_stems['rate']

    stems = ('drums', 'bass', 'vocals', 'other')

    for s in stems:
        write(f'output/{s}.wav', rate, separated_stems[s])

def write_dict_to_zip_in_memory(separated_stems):
    rate = separated_stems['rate']
    stems = ('drums', 'bass', 'vocals', 'other')

    zip_in_memory = io.BytesIO()
    with zipfile.ZipFile(zip_in_memory, 'w', zipfile.ZIP_DEFLATED) as zf:
        for s in stems:
            wav_file_in_memory = io.BytesIO()
            write(wav_file_in_memory, rate, separated_stems[s])
            wav_file_in_memory.seek(0)

            zf.writestr(f'{s}.wav', wav_file_in_memory.getvalue())

    zip_in_memory.seek(0)
    return zip_in_memory


to_files = True

if __name__ == '__main__':
    if to_files is True:
        #este ok asa pentru ca daca citesc 'fisiere' din memorie o sa am functii care folosesc io.BytesIO()
        #si scipty.io.wavfile
        rate, song = read(f'input/Little Sister.wav')

        write_dict_to_files(separate_into_dict(song, rate=rate))
