import librosa
import numpy as np
from basis import ScoreBasis

class FWSEGSNR(ScoreBasis):
    def __init__(self):
        super(FWSEGSNR, self).__init__(name='FWSEGSNR')
        self.intrusive = False

    def windowed_scoring(self, audios, score_rate):
        if len(audios) != 2:
            raise ValueError('FWSEGSNR needs a reference and a test signals.')
        return fwsegsnr(audios[1], audios[0], score_rate)

def fwsegsnr(x, y, fs, frame_sz = 0.025, shift_sz= 0.01, win='hann', numband=23):
    epsilon = np.finfo(np.float32).eps
    frame = int(np.fix(frame_sz * fs))
    shift = int(np.fix(shift_sz * fs))
    window = win
    nband = numband
    noverlap = frame - shift
    fftpt = int(2**np.ceil(np.log2(np.abs(frame))))
    x = x / np.sqrt(sum(np.power(x, 2)))
    y = y / np.sqrt(sum(np.power(y, 2)))

    assert len(x) == len(y), print('Wav length are not matched!')
    X_stft = np.abs(librosa.stft(x, n_fft=fftpt, hop_length=shift, win_length=frame, window=window, center=False))
    Y_stft = np.abs(librosa.stft(y, n_fft=fftpt, hop_length=shift, win_length=frame, window=window, center=False))

    num_freq = X_stft.shape[0]
    num_frame = X_stft.shape[1]

    X_mel = librosa.feature.melspectrogram(S=X_stft, sr=fs, n_mels=nband, fmin=0, fmax=fs/2)
    Y_mel = librosa.feature.melspectrogram(S=Y_stft, sr=fs, n_mels=nband, fmin=0, fmax=fs/2)

    # Calculate SNR.

    W = np.power(Y_mel, 0.2)
    E = X_mel - Y_mel
    E[E == 0.0] = epsilon
    E_power = np.power(E, 2)
    Y_div_E = np.divide((np.power(Y_mel,2)), (np.power(E,2)))
    Y_div_E[Y_div_E==0] = epsilon
    ds = 10 * np.divide(np.sum(np.multiply(W, np.log10(Y_div_E)), 1), np.sum(W, 1))
    ds[ds > 35] = 35
    ds[ds < -10] = -10
    d = np.mean(ds)
    return d

