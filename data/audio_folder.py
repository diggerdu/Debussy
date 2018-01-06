import os
import os.path
import librosa
import soundfile as sf
import librosa as lb
import numpy as np
import shutil
import random

AUDIO_EXTENSIONS = [
    '.wav',
    '.WAV',
]

def is_audio_file(filename):
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)

def getMel(path, opt):
    try:
        wav, sr = sf.read(path, dtype='float32')
    except:
        assert False
    try:
        assert sr == opt.SR
    except AssertionError:
        wav = librosa.resample(wav, sr, opt.SR)
    print('{}:{}'.format(path, wav.shape[0]))
    target_len = opt.len - 512

    if wav.shape[0] > target_len:
        offsetSum = 0
        maxSum = 0
        maxHead = 0
        for head in range(wav.shape[0] - target_len):
            if offsetSum > maxSum:
                offsetSum = maxSum
                maxHead = head
            offsetSum -= np.abs(wav[head])
            offsetSum += np.abs(wav[head + target_len])
        print('maxHead: ', maxHead)

        wav = wav[maxHead:maxHead + target_len]

    if wav.shape[0] < target_len:
        ExtraLen = target_len - wav.shape[0]
        PrevExtraLen = np.random.randint(ExtraLen)
        PostExtraLen = ExtraLen - PrevExtraLen
        PrevExtra = np.zeros((PrevExtraLen, ), dtype=np.float32)
        PostExtra = np.zeros((PostExtraLen, ), dtype=np.float32)
        wav = np.concatenate((PrevExtra, wav, PostExtra))

    assert wav.shape[0] == target_len

    wav = wav - np.mean(wav)
    if np.max(np.abs(wav)) > 0:
        wav = wav / np.max(np.abs(wav))
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
    return melsp.astype(np.float32)



def loadData(Dir, opt):
    assert os.path.isdir(Dir), '%s is not a valid directory' % dir
    audios = list()
    labels = list()
    fnames = list()

    for root, _, fns in sorted(os.walk(Dir)):
        for fname in fns:
            if is_audio_file(fname):
                path = os.path.join(root, fname)
                label = path.split('/')[-2]

                melsp = getMel(path, opt)

                audios.append(melsp)
                labels.append(label)
                fnames.append(fname)


                # TODO
                if opt.isTrain:
                    os.system("sox {0} /tmp/outTempo1.9.wav tempo -s 1.9".format(path))
                    melsp = getMel('/tmp/outTempo1.9.wav', opt)
                    audios.append(melsp)
                    labels.append(label)
                    fnames.append(fname + '_tempo1.9')

                    #os.system("sox {0} /tmp/outTempo0.75.wav tempo -s 0.75 silence 1 0.1 1% -1 0.1 1%".format(path))
                    os.system("sox {0} /tmp/outTempo0.75.wav tempo -s 0.75".format(path))
                    melsp = getMel('/tmp/outTempo0.75.wav', opt)
                    audios.append(melsp)
                    labels.append(label)
                    fnames.append(fname + '_tempo0.75')

                    os.system("sox {0} /tmp/outPitch500.wav pitch 500".format(path))
                    melsp = getMel('/tmp/outPitch500.wav', opt)
                    audios.append(melsp)
                    labels.append(label)
                    fnames.append(fname + '_pitch500')

                    os.system("sox {0} /tmp/outPitch-500.wav pitch -500".format(path))
                    melsp = getMel('/tmp/outPitch-500.wav', opt)
                    audios.append(melsp)
                    labels.append(label)
                    fnames.append(fname + '_pitch-500')


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
