from basis import ScoreBasis
import numpy as np
from numpy.linalg import norm

class SISDR(ScoreBasis):
    def __init__(self):
        super(SISDR, self).__init__(name='SISDR')
        self.intrusive = False

    def windowed_scoring(self, audios, score_rate):
        # as provided by @Jonathan-LeRoux and slightly adapted for the case of just one reference
        # and one estimate.
        # see original code here: https://github.com/sigsep/bsseval/issues/3#issuecomment-494995846
        if len(audios) != 2:
            raise ValueError('PESQ needs a reference and a test signals.')
        eps = np.finfo(audios[0].dtype).eps
        reference = audios[1].reshape(audios[1].size, 1)
        estimate = audios[0].reshape(audios[0].size, 1)
        
        Rss = np.dot(reference.T, reference)

        # get the scaling factor for clean sources
        a = (eps + np.dot(reference.T, estimate)) / (Rss + eps)

        e_true = a * reference
        e_res = estimate - e_true

        Sss = (e_true**2).sum()
        Snn = (e_res**2).sum()

        return 10 * np.log10((eps+ Sss)/(eps + Snn))

