import SourceX
import torch
from scipy.io.wavfile import read, write
import numpy
import random
import gc

if __name__ == '__main__':
    with torch.no_grad():
        dtype = torch.float32
        device = torch.device("cuda")
        #print(torch.cuda.is_available())
        torch.set_default_device("cuda")
        model = SourceX.AudioModel(torch.nn.Tanh(), torch.nn.Mish(), torch.nn.ReLU())
        model.to(device='cuda')
        try:
            checkpoint = torch.load(f'istorie antrenari/azi/model.model', weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])

            model.eval()
            print('am incarcat model.model')
        except:
            print('nu am incarcat nici-un state dict')

        rate, input_file = read(f'input/input.wav')
        input_file = input_file / numpy.iinfo(numpy.int16).max
        print(input_file)

        # genereaza batches
        # posibil memory leak
        x_batches = []
        total_batched = 0
        batch_size = 1323000  # 3 - 30 secunde
        while total_batched < input_file.shape[0]:
            if input_file.shape[0] - batch_size >= total_batched:
                x_batches.append(input_file[total_batched: total_batched + batch_size, :])
                total_batched += batch_size
            else:
                x_batches.append(input_file[total_batched:, :])
                break

        y_pred = None
        for x_batch in x_batches:
            print('started batch')
            x_true = torch.from_numpy(SourceX.genereaza_tensor_din_stereo(x_batch))
            x_true = x_true.to(torch.float32)
            x_true = x_true.to(device="cuda")
            output = model(x_true)

            #s a intamplat ceva funky si acum sunt speriat
            del x_true
            output = output.to(device='cpu')
            if y_pred == None:
                y_pred = output
            else:
                y_pred = torch.cat([y_pred, output], dim=1)

            del output

        y_pred_np = y_pred.detach().numpy()

        write(f'output/drums.wav', 44100, (y_pred_np[0, :, :] * 32767).astype(numpy.int16))
        write(f'output/bass.wav', 44100, (y_pred_np[1, :, :] * 32767).astype(numpy.int16))
        write(f'output/vocals.wav', 44100, (y_pred_np[2, :, :] * 32767).astype(numpy.int16))
        write(f'output/other.wav', 44100, (y_pred_np[3, :, :] * 32767).astype(numpy.int16))

        del y_pred
        del model

        gc.collect()
        torch.cuda.empty_cache()
