import numpy as np
from pesq import pesq

from vyvodata.tools.speechscore.basis import ScoreBasis
from vyvodata.tools.speechscore.scores.helper import norm_mos, ssnr, wss


class CBAK(ScoreBasis):
    def __init__(self):
        super().__init__(name="CBAK")
        self.score_rate = 16000
        self.intrusive = False

    def windowed_scoring(self, audios, score_rate):
        if len(audios) != 2:
            raise ValueError("CBAK needs a reference and a test signals.")
        return cal_cbak(audios[0], audios[1], score_rate)


def cal_cbak(target_wav, pred_wav, fs):
    alpha = 0.95

    # Compute WSS measure
    wss_dist_vec = wss(target_wav, pred_wav, fs)
    wss_dist_vec = sorted(wss_dist_vec, reverse=False)
    wss_dist = np.mean(wss_dist_vec[: int(round(len(wss_dist_vec) * alpha))])

    # Compute the SSNR
    snr_mean, seg_snr_mean = ssnr(target_wav, pred_wav, fs)
    seg_snr = np.mean(seg_snr_mean)

    # Compute the PESQ
    pesq_raw = pesq(fs, target_wav, pred_wav, "wb")

    # cbak
    cbak = 1.634 + 0.478 * pesq_raw - 0.007 * wss_dist + 0.063 * seg_snr
    cbak = norm_mos(cbak)

    return cbak
