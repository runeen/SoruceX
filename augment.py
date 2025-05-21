import random
import copy
import time
#import numpy
import torch
from sympy import false


class ChannelWiseLinearTransform:
    def __init__(self, mono=False):
        self.mono = mono

    def forward(self, y_true):
        if self.mono:
            y_output = copy.deepcopy(y_true)

            y_output[:, :, 0] = y_true[:, :, 0] * 0.5 + y_output[:, :, 1] * 0.5
            y_output[:, :, 1] = y_true[:, :, 0] * 0.5 + y_output[:, :, 1] * 0.5
        else:
            #alegem a si aplha
            a = random.uniform(0, 0.9)
            alph = random.uniform(0, 0.9)

            #alegem b si beta
            b = random.uniform(0, 1.0 - a)
            beta = random.uniform(0, 1.0 - alph)

            y_output = copy.deepcopy(y_true)

            y_output[:, :, 0] = y_true[:, :, 0] * a + y_output[:, :, 1] * b
            y_output[:, :, 1] = y_true[:, :, 0] * alph + y_output[:, :, 1] * beta

        return y_output

class MuteStems:
    def forward(self, y_true):
        #alegem ce stem sa fie muted
        a = random.choice([0, 1])
        b = random.choice([0, 1])
        c = random.choice([0, 1])
        d = random.choice([0, 1])

        y_output = copy.deepcopy(y_true)

        y_output[0, :, :] = y_true[0, :, :] * a
        y_output[1, :, :] = y_true[1, :, :] * b
        y_output[2, :, :] = y_true[2, :, :] * c
        y_output[3, :, :] = y_true[3, :, :] * d

        return y_output


class StemReLeveling:
    def forward(self, y_true):
        #alegem cu cat sa coboram fiecare stem
        a = random.uniform(0, 1)
        b = random.uniform(0, 1)
        c = random.uniform(0, 1)
        d = random.uniform(0, 1)

        y_output = copy.deepcopy(y_true)

        y_output[0, :, :] = y_true[0, :, :] * a
        y_output[1, :, :] = y_true[1, :, :] * b
        y_output[2, :, :] = y_true[2, :, :] * c
        y_output[3, :, :] = y_true[3, :, :] * d

        return y_output




class Augment:
    def __init__(self):
        self.channel_lin_tran = ChannelWiseLinearTransform()
        self.mono = ChannelWiseLinearTransform(mono=True)
        self.re_level = StemReLeveling()
        self.mute = MuteStems()
        self.x_true = None
        self.y_true = None
        self.modified = False


    def calc_x_true(self, rand_amp = False) -> int:
        '''
        Calculeaza x_true si normalizeaza ambii tensori daca
        depaseste 1
        :return:
        '''

        self.x_true = torch.sum(self.y_true, dim = 0)

        max_amp_x = torch.max(torch.abs(self.x_true))
        if max_amp_x < 0.001:
            return 1
        if max_amp_x < 0.90 or max_amp_x > 1:
            self.y_true = self.y_true / max_amp_x
            self.x_true = torch.sum(self.y_true, dim = 0)

        if rand_amp:
            rand = random.uniform(0.5, 1)
            self.y_true = self.y_true * rand

        return 0

    def replace_y_true_with_diff(self):
        '''
        calculeaza diferenta intre stemurile introduse la inceputul augemntarii
        si ce stem-uri avem in self.y_true si o stocheaza in  self.y_true
        :return:
        '''
        if self.modified:
            self.y_true = self.y_original - self.y_true



    def forward(self, y_true):
        self.y_true = y_true
        self.y_original = copy.deepcopy(self.y_true)

        if random.uniform(0, 3) > 2 :
            self.y_true = self.re_level.forward(self.y_true)
            self.modified = True
        if random.uniform(0, 3) > 2:
            self.y_true = self.mute.forward(self.y_true)
            self.modified = True
        if random.uniform(0, 11) > 10 :
            self.y_true = self.channel_lin_tran.forward(self.y_true)
            self.modified = True
        elif random.uniform(0, 11) > 10:
            self.y_true = self.mono.forward(self.y_true)
            self.modified = True
        if random.uniform(0, 13) > 12 : self.replace_y_true_with_diff()

        err = self.calc_x_true()

        return self.y_true, self.x_true, err

if __name__ == '__main__':
    import musdb
    from scipy.io.wavfile import write
    #testam modulul augment

    mus = musdb.DB(subsets="train", split='valid')

    rate = 44100

    aug = Augment()
    t0 = time.perf_counter()
    input_file = mus[10].audio
    stems = mus[10].stems[(1, 2, 4, 3), :, :]
    y_true, x_true, err = aug.forward(stems)
    print(f'time to augment: {time.perf_counter() - t0}')



    diff_np = x_true - input_file

    #print(numpy.max(abs(x_true)))


    #write(f'output/aug_test_x_true.wav', rate, (x_true *  32767).astype(numpy.int16))
    #write(f'output/aud_test_diff.wav', rate, (diff_np *  32767).astype(numpy.int16))





