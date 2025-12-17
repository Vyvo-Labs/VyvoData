from basis import ScoreBasis


class SRMR(ScoreBasis):
    def __init__(self):
        super(SRMR, self).__init__(name='SRMR')
        self.intrusive = True
        self.score_rate = 16000

    def windowed_scoring(self, audios, score_rate):
        from scores.srmr.cal_srmr import cal_SRMR
        return cal_SRMR(audios[0], score_rate, n_cochlear_filters=23,low_freq=125, min_cf=4,max_cf=128, fast=True, norm=False)[0]

