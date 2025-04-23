import torch
import random
import copy


class ChannelWiseLinearTransform(torch.nn.Module):
    def __init__(self, mono=False):
        super().__init__()
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

class StemReLeveling(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_true):
        #alegem cu cat sa coboram fiecare stem
        a = random.uniform(0, 1)
        #print(a)
        b = random.uniform(0, 1)
        #print(b)
        c = random.uniform(0, 1)
        #print(c)
        d = random.uniform(0, 1)
        #print(d)

        y_output = copy.deepcopy(y_true)

        y_output[0, :, :] = y_true[0, :, :] * a
        y_output[1, :, :] = y_true[1, :, :] * b
        y_output[2, :, :] = y_true[2, :, :] * c
        y_output[3, :, :] = y_true[3, :, :] * d

        return y_output




class Augment(torch.nn.Module):
    def __init__(self, y_true):
        super().__init__()
        self.channel_lin_tran = ChannelWiseLinearTransform()
        self.mono = ChannelWiseLinearTransform(mono=True)
        self.re_level = StemReLeveling()
        self.x_true = None
        self.y_true = y_true
        self.modified = False


    def calc_x_true(self):
        '''
        Calculeaza x_true si normalizeaza ambii tensori daca
        depaseste 1
        :return:
        '''

        self.x_true = torch.sum(self.y_true, dim = 0)

        max_amp_x = torch.max(torch.abs(self.x_true))

        if max_amp_x > 1.:
            self.y_true = self.y_true - self.y_true * (max_amp_x - 1)
            self.x_true = torch.sum(self.y_true, dim = 0)
        if max_amp_x < 1.:
            self.y_true = self.y_true + self.y_true * (1 - max_amp_x)
            self.x_true = torch.sum(self.y_true, dim = 0)

    def replace_y_true_with_diff(self):
        '''
        calculeaza diferenta intre stemurile introduse la inceputul augemntarii
        si ce stem-uri avem in self.y_true si o stocheaza in  self.y_true
        :return:
        '''
        if self.modified:
            self.y_true = self.y_original - self.y_true



    def forward(self):
        self.y_original = copy.deepcopy(self.y_true)

        if random.uniform(0, 3) > 2 :
            self.y_true = self.re_level(self.y_true)
            self.modified = True
        if random.uniform(0, 11) > 10 :
            self.y_true = self.channel_lin_tran(self.y_true)
            self.modified = True
        elif random.uniform(0, 11) > 10:
            self.y_true = self.mono(self.y_true)
            self.modified = True
        if random.uniform(0, 13) > 12 : self.replace_y_true_with_diff()

        self.calc_x_true()

        return self.y_true, self.x_true

if __name__ == '__main__':
    import musdb
    from scipy.io.wavfile import write
    import numpy
    #testam modulul augment

    mus = musdb.DB(subsets="train", split='valid')

    rate = 44100
    input_file = mus[10].audio
    stems = torch.from_numpy(mus[10].stems[(1, 2, 4, 3), :, :])
    #og_stems = copy.deepcopy(stems)

    aug = Augment(stems)

    y_true, x_true = aug()

    y_true_np = y_true.numpy()
    x_true_np = x_true.numpy()

    diff_np = x_true_np - input_file


    write(f'output/aug_test_x_true.wav', rate, (x_true_np *  32767).astype(numpy.int16))
    write(f'output/aud_test_diff.wav', rate, (diff_np *  32767).astype(numpy.int16))

    '''
    aug = Augment(stems)

    y_true, x_true = aug()

    #y_true_np = y_true.numpy()
    x_true_np = x_true.numpy()
    write(f'output/aug_test_x_true_mono.wav', rate, (x_true_np *  32767).astype(numpy.int16))
    '''





