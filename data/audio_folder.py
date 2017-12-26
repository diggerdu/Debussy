import os
import os.path
import librosa
import soundfile as sf
import librosa as lb
import numpy as np
import shutil

AUDIO_EXTENSIONS = [
    '.wav',
    '.WAV',
]

def is_audio_file(filename):
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)


def loadData(Dir, opt):
    assert os.path.isdir(Dir), '%s is not a valid directory' % dir
    audios = list()
    labels = list()
    fnames = list()

    for root, _, fns in sorted(os.walk(Dir)):
        for fname in fns:
            if is_audio_file(fname):
                path = os.path.join(root, fname)
                print(path)
                label = path.split('/')[-2]
                try:
                    wav, sr = sf.read(path, dtype='float32')
                except:
                    continue
                try:
                    assert sr == opt.SR
                except AssertionError:
                    wav = librosa.resample(wav, sr, opt.SR)

                target_len = opt.len - 512

                if wav.shape[0] >= target_len:
                    head = random.randint(0, wav.shape[0] - target_len)
                    wav = wav[head:head + target_len]
                if wav.shape[0] < target_len:
                    ExtraLen = target_len - wav.shape[0]
                    PrevExtraLen = np.random.randint(ExtraLen)
                    PostExtraLen = ExtraLen - PrevExtraLen
                    PrevExtra = np.zeros((PrevExtraLen, ), dtype=np.float32)
                    PostExtra = np.zeros((PostExtraLen, ), dtype=np.float32)
                    wav = np.concatenate((PrevExtra, wav, PostExtra))

                wav = wav - np.mean(wav)
                if np.max(np.abs(wav)) > 0:
                    wav = wav / np.max(np.abs(wav))
                # sf.write(path, wav, opt.SR)

                melsp = librosa.feature.melspectrogram(
                    y=wav,
                    sr=sr,
                    S=None,
                    n_fft=opt.nfft,
                    hop_length=opt.hop,
                    power=2.0,
                    n_mels=64,
                    fmax=sr // 2)

                # TODO
                eps = 1e-3
                melsp = np.log(melsp + eps)

                audios.append(melsp.astype(np.float32))
                labels.append(label)
                fnames.append(fname)
    return {'audios':audios, 'labels':labels, 'fnames':fnames}



def make_dataset(opt):
    audios = []
    labels = []
    fnames = []
    try:
        audios = np.load(opt.dumpPath + "/audios.npy")
        labels = np.load(opt.dumpPath + "/labels.npy").tolist()
        fnames = np.load(opt.dumpPath + "/fnames.npy").tolist()
        print("######CAUTION:previous generated dataset loaded###########")
        return audios, labels, fnames
    except:
        pass
    try:
        shutil.rmtree(opt.dumpPath)
    except:
        pass
    os.mkdir(opt.dumpPath)

    data = loadData(opt.Path, opt)
    audios = data['audios']
    labels = data['labels']
    fnames = data['fnames']
    if opt.additionPath is not None:
        additionData = loadData(opt.additionPath, opt)
        audios.extend(additionData['audios'])
        labels.extend(additionData['labels'])
        fnames.extend(additionData['fnames'])
    np.save(opt.dumpPath + "/audios.npy", np.array(audios))
    np.save(opt.dumpPath + "/labels.npy", np.array(labels))
    np.save(opt.dumpPath + "/fnames.npy", np.array(fnames))
    return audios, labels, fnames
