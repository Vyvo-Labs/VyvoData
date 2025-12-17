from basis import ScoreBasis
import numpy as np
from pesq import pesq
from scores.helper import wss, llr, SSNR, norm_mos

class COVL(ScoreBasis):
    def __init__(self):
        super(COVL, self).__init__(name='COVL')
        self.score_rate = 16000
        self.intrusive = False

    def windowed_scoring(self, audios, score_rate):
        if len(audios) != 2:
            raise ValueError('COVL needs a reference and a test signals.')
        return cal_COVL(audios[0], audios[1], score_rate)

def cal_COVL(target_wav, pred_wav, fs):
    alpha   = 0.95

    # Compute WSS measure
    wss_dist_vec = wss(target_wav, pred_wav, fs)
    wss_dist_vec = sorted(wss_dist_vec, reverse=False)
    wss_dist     = np.mean(wss_dist_vec[:int(round(len(wss_dist_vec) * alpha))])

    # Compute LLR measure
    LLR_dist = llr(target_wav, pred_wav, fs)
    LLR_dist = sorted(LLR_dist, reverse=False)
    LLRs     = LLR_dist
    LLR_len  = round(len(LLR_dist) * alpha)
    llr_mean = np.mean(LLRs[:LLR_len])

    # Compute the PESQ
    pesq_raw = pesq(fs, target_wav, pred_wav, 'wb')

    # Covl
    Covl = 1.594 + 0.805 * pesq_raw - 0.512 * llr_mean - 0.007 * wss_dist
    Covl = norm_mos(Covl)

    return Covl
