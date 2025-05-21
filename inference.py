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

        rate, input_file = read(f'input/Little Sister.wav')
        input_file = input_file / numpy.iinfo(numpy.int16).max
        print(input_file)

        # genereaza batches
        # posibil memory leak
        x_batches = []
        total_batched = 0
        batch_size = 2646000  # 3 - 30 secunded
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
            x_batch = torch.tensor(x_batch)
            x_batch = x_batch.to(torch.float32)
            x_batch = x_batch.to(device="cuda")
            output = model(x_batch)

            #s a intamplat ceva funky si acum sunt speriat
            del x_batch
            output = output.to(device='cpu')
            if y_pred == None:
                y_pred = output
            else:
                y_pred = torch.cat([y_pred, output], dim=1)

            del output

        y_pred_np = y_pred.detach().numpy()
        print(y_pred_np)

        write(f'output/drums.wav', rate, (y_pred_np[(0, 1), :] * 32767).T.astype(numpy.int16))
        write(f'output/bass.wav', rate, (y_pred_np[(2, 3), :] * 32767).T.astype(numpy.int16))
        write(f'output/vocals.wav', rate, (y_pred_np[(4, 5), :] * 32767).T.astype(numpy.int16))
        write(f'output/other.wav', rate, (y_pred_np[(6, 7), :] * 32767).T.astype(numpy.int16))

        del y_pred
        del model

        gc.collect()
        torch.cuda.empty_cache()
