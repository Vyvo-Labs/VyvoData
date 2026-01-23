import numpy as np
from pesq import pesq

from vyvodata.tools.speechscore.basis import ScoreBasis
from vyvodata.tools.speechscore.scores.helper import llr, norm_mos, wss


class COVL(ScoreBasis):
    def __init__(self):
        super().__init__(name="COVL")
        self.score_rate = 16000
        self.intrusive = False

    def windowed_scoring(self, audios, score_rate):
        if len(audios) != 2:
            raise ValueError("COVL needs a reference and a test signals.")
        return cal_covl(audios[0], audios[1], score_rate)


def cal_covl(target_wav, pred_wav, fs):
    alpha = 0.95

    # Compute WSS measure
    wss_dist_vec = wss(target_wav, pred_wav, fs)
    wss_dist_vec = sorted(wss_dist_vec, reverse=False)
    wss_dist = np.mean(wss_dist_vec[: int(round(len(wss_dist_vec) * alpha))])

    # Compute LLR measure
    llr_dist = llr(target_wav, pred_wav, fs)
    llr_dist = sorted(llr_dist, reverse=False)
    llrs = llr_dist
    llr_len = round(len(llr_dist) * alpha)
    llr_mean = np.mean(llrs[:llr_len])

    # Compute the PESQ
    pesq_raw = pesq(fs, target_wav, pred_wav, "wb")

    # covl
    covl = 1.594 + 0.805 * pesq_raw - 0.512 * llr_mean - 0.007 * wss_dist
    covl = norm_mos(covl)

    return covl
