from basis import ScoreBasis
import numpy as np
from scipy.linalg import toeplitz
from scores.helper import lpcoeff

class LLR(ScoreBasis):
    def __init__(self):
        super(LLR, self).__init__(name='LLR')
        self.intrusive = False

    def windowed_scoring(self, audios, score_rate):
        if len(audios) != 2:
            raise ValueError('LLR needs a reference and a test signals.')
        return cal_LLR(audios[0], audios[1], score_rate)

def cal_LLR(ref_wav, deg_wav, srate):
    # obtained from https://github.com/wooseok-shin/MetricGAN-plus-pytorch/blob/main/metric_functions/metric_helper.py
    clean_speech = ref_wav
    processed_speech = deg_wav
    clean_length = ref_wav.shape[0]
    processed_length = deg_wav.shape[0]
    assert clean_length == processed_length, clean_length

    winlength = round(30 * srate / 1000.) # 240 wlen in samples
    skiprate = np.floor(winlength / 4)
    if srate < 10000:
        # LPC analysis order
        P = 10
    else:
        P = 16

    # For each frame of input speech, calculate the Log Likelihood Ratio
    num_frames = int(clean_length / skiprate - (winlength / skiprate))
    start = 0
    time = np.linspace(1, winlength, winlength) / (winlength + 1)
    window = 0.5 * (1 - np.cos(2 * np.pi * time))
    distortion = []

    for frame_count in range(num_frames):
        # (1) Get the Frames for the test and reference speeech.
        # Multiply by Hanning window.
        clean_frame = clean_speech[start:start+winlength]
        processed_frame = processed_speech[start:start+winlength]
        clean_frame = clean_frame * window
        processed_frame = processed_frame * window

        # (2) Get the autocorrelation logs and LPC params used
        # to compute the LLR measure
        R_clean, Ref_clean, A_clean = lpcoeff(clean_frame, P)
        R_processed, Ref_processed, A_processed = lpcoeff(processed_frame, P)
        A_clean = A_clean[None, :]
        A_processed = A_processed[None, :]

        # (3) Compute the LLR measure
        numerator = A_processed.dot(toeplitz(R_clean)).dot(A_processed.T)
        denominator = A_clean.dot(toeplitz(R_clean)).dot(A_clean.T)

        if (numerator/denominator) <= 0:
            continue

        log_ = np.log(numerator / denominator)
        distortion.append(np.squeeze(log_))
        start += int(skiprate)
    return np.mean(np.nan_to_num(np.array(distortion)))

