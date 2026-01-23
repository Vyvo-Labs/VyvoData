import numpy as np
from pesq import pesq

from vyvodata.tools.speechscore.basis import ScoreBasis
from vyvodata.tools.speechscore.scores.helper import llr, norm_mos, wss


class CSIG(ScoreBasis):
    def __init__(self):
        super().__init__(name="CSIG")
        self.score_rate = 16000

    def windowed_scoring(self, audios, score_rate):
        if len(audios) != 2:
            raise ValueError("CSIG needs a reference and a test signals.")
        return cal_csig(audios[0], audios[1], score_rate)


def cal_csig(target_wav, pred_wav, fs):
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

    # csig
    csig = 3.093 - 1.029 * llr_mean + 0.603 * pesq_raw - 0.009 * wss_dist
    csig = float(norm_mos(csig))

    return csig
