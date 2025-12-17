from basis import ScoreBasis
import numpy as np
from pesq import pesq
from scores.helper import wss, llr, SSNR, norm_mos

class CBAK(ScoreBasis):
    def __init__(self):
        super(CBAK, self).__init__(name='CBAK')
        self.score_rate = 16000
        self.intrusive = False

    def windowed_scoring(self, audios, score_rate):
        if len(audios) != 2:
            raise ValueError('CBAK needs a reference and a test signals.')
        return cal_CBAK(audios[0], audios[1], score_rate)

def cal_CBAK(target_wav, pred_wav, fs):
    alpha   = 0.95

    # Compute WSS measure
    wss_dist_vec = wss(target_wav, pred_wav, fs)
    wss_dist_vec = sorted(wss_dist_vec, reverse=False)
    wss_dist     = np.mean(wss_dist_vec[:int(round(len(wss_dist_vec) * alpha))])

    # Compute the SSNR
    snr_mean, segsnr_mean = SSNR(target_wav, pred_wav, fs)
    segSNR = np.mean(segsnr_mean)

    # Compute the PESQ
    pesq_raw = pesq(fs, target_wav, pred_wav, 'wb')

    # Cbak
    Cbak = 1.634 + 0.478 * pesq_raw - 0.007 * wss_dist + 0.063 * segSNR
    Cbak = norm_mos(Cbak)

    return Cbak

