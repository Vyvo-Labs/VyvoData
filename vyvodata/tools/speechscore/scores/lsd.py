from basis import ScoreBasis
import numpy as np
import librosa

EPS = 1e-12

class LSD(ScoreBasis):
    def __init__(self):
        super(LSD, self).__init__(name='LSD')
        self.intrusive = False
        self.mono = True

    def windowed_scoring(self, audios, score_rate):
        if len(audios) != 2:
            raise ValueError('LSD needs a reference and a test signals.')
        est = wav_to_spectrogram(audios[1], score_rate)
        target = wav_to_spectrogram(audios[0], score_rate)
        return cal_LSD(est, target)

def wav_to_spectrogram(wav, rate):
    hop_length = int(rate / 100)
    n_fft = int(2048 / (48000 / rate)) 
    spec = np.abs(librosa.stft(wav, hop_length=hop_length, n_fft=n_fft))
    spec = np.transpose(spec, (1, 0))
    return spec

def cal_LSD(est, target):
    log_ratio = np.log10(target**2 / ((est + EPS) ** 2) + EPS) ** 2
    lsd_ = np.mean(np.mean(log_ratio, axis=1) ** 0.5, axis=0)
    return lsd_
