import librosa
import numpy as np

from vyvodata.tools.speechscore.basis import ScoreBasis


class FWSEGSNR(ScoreBasis):
    def __init__(self):
        super().__init__(name="FWSEGSNR")
        self.intrusive = False

    def windowed_scoring(self, audios, score_rate):
        if len(audios) != 2:
            raise ValueError("FWSEGSNR needs a reference and a test signals.")
        return fwsegsnr(audios[1], audios[0], score_rate)


def fwsegsnr(x, y, fs, frame_sz=0.025, shift_sz=0.01, win="hann", numband=23):
    epsilon = np.finfo(np.float32).eps
    frame = int(np.fix(frame_sz * fs))
    shift = int(np.fix(shift_sz * fs))
    window = win
    nband = numband
    fftpt = int(2 ** np.ceil(np.log2(np.abs(frame))))
    x = x / np.sqrt(sum(np.power(x, 2)))
    y = y / np.sqrt(sum(np.power(y, 2)))

    assert len(x) == len(y), print("Wav length are not matched!")
    x_stft = np.abs(
        librosa.stft(
            x,
            n_fft=fftpt,
            hop_length=shift,
            win_length=frame,
            window=window,
            center=False,
        )
    )
    y_stft = np.abs(
        librosa.stft(
            y,
            n_fft=fftpt,
            hop_length=shift,
            win_length=frame,
            window=window,
            center=False,
        )
    )

    x_mel = librosa.feature.melspectrogram(
        S=x_stft, sr=fs, n_mels=nband, fmin=0, fmax=fs / 2
    )
    y_mel = librosa.feature.melspectrogram(
        S=y_stft, sr=fs, n_mels=nband, fmin=0, fmax=fs / 2
    )

    # Calculate SNR.

    w = np.power(y_mel, 0.2)
    e = x_mel - y_mel
    e[e == 0.0] = epsilon
    y_div_e = np.divide((np.power(y_mel, 2)), (np.power(e, 2)))
    y_div_e[y_div_e == 0] = epsilon
    ds = 10 * np.divide(np.sum(np.multiply(w, np.log10(y_div_e)), 1), np.sum(w, 1))
    ds[ds > 35] = 35
    ds[ds < -10] = -10
    d = np.mean(ds)
    return d
