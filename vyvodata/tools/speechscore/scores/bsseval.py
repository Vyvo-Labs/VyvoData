import numpy as np
from basis import ScoreBasis


class BSSEval(ScoreBasis):
    def __init__(self):
        super(BSSEval, self).__init__(name='BSSEval')
        self.intrusive = False

    def windowed_scoring(self, audios, score_rate):
        bss_window = np.inf
        bss_hop = np.inf
        from museval.metrics import bss_eval
        if len(audios) != 2:
            raise ValueError('BSSEval needs a reference and a test signals.')
        
        result = bss_eval(reference_sources=audios[1][None,...], # shape: [nsrc, nsample, nchannels]
                        estimated_sources=audios[0][None,...],
                        window=bss_window * score_rate,
                        hop=bss_hop * score_rate)
        return {'SDR': result[0][0][0], 'ISR': result[1][0][0], 'SAR': result[3][0][0]}
