import os.path
from data.base_dataset import BaseDataset
from data.audio_folder import make_dataset
import librosa
import soundfile as sf
import numpy as np
import random


class AudioDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.Dir = opt.Path
        self.Data, self.Labels, self.Fnames = make_dataset(self.Dir, opt)

        self.SR = opt.SR
        self.hop = opt.hop
        self.nfft = self.opt.nfft
        self.table = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence', 'unknown']

    def __getitem__(self, index):
        Data = self.Data[index]
        Label = self.Labels[index]
        fname = self.Fnames[index]
        Audio = np.expand_dims(Data, axis=0)
        # Audio = self.load_audio(Data)

        assert Audio.dtype==np.float32

        try:
            LabelCode = self.table.index(Label)
        except:
            LabelCode = len(self.table) - 1
        return {
        'Audio': Audio,
        'Label': LabelCode,
        'Fname': fname}

    def __len__(self):
        # return len(self.FilesClean)
        return len(self.Data)
        # return max(len(self.Clean), len(self.Noise))

    def name(self):
        return "AudioDataset"

    def load_audio(self, data):
        target_len = self.opt.len
        if data.shape[0] >= target_len:
            head = random.randint(0, data.shape[0] - target_len)
            data = data[head:head + target_len]
        if data.shape[0] < target_len:
            ExtraLen = target_len - data.shape[0]
            PrevExtraLen = np.random.randint(ExtraLen)
            PostExtraLen = ExtraLen - PrevExtraLen
            PrevExtra = np.zeros((PrevExtraLen, ), dtype=np.float32)
            PostExtra = np.zeros((PostExtraLen, ), dtype=np.float32)
            data = np.concatenate((PrevExtra, data, PostExtra))

        data = data - np.mean(data)
        assert data.dtype == np.float32

        assert data.shape[0] == self.opt.len
        return data
