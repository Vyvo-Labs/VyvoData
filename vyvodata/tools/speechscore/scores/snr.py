from basis import ScoreBasis
import numpy as np

class SNR(ScoreBasis):
    def __init__(self):
        super(SNR, self).__init__(name='SNR')
        self.intrusive = False

    def windowed_scoring(self, audios, score_rate):
        if len(audios) != 2:
            raise ValueError('SNR needs a reference and a test signals.')
        return cal_SNR(audios[0], audios[1], score_rate)

def cal_SNR(ref_wav, deg_wav, srate=16000, eps=1e-10):
    # obtained from https://github.com/wooseok-shin/MetricGAN-plus-pytorch/blob/main/metric_functions/metric_helper.py
    """ Segmental Signal-to-Noise Ratio Objective Speech Quality Measure
        This function implements the segmental signal-to-noise ratio
        as defined in [1, p. 45] (see Equation 2.12).
    """
    clean_speech     = ref_wav
    processed_speech = deg_wav
    clean_length     = ref_wav.shape[0]
    processed_length = deg_wav.shape[0]
    
    # scale both to have same dynamic range. Remove DC too.
    clean_speech     -= clean_speech.mean()
    processed_speech -= processed_speech.mean()
    processed_speech *= (np.max(np.abs(clean_speech)) / np.max(np.abs(processed_speech)))
   
    # Signal-to-Noise Ratio 
    dif = ref_wav - deg_wav
    overall_snr = 10 * np.log10(np.sum(ref_wav ** 2) / (np.sum(dif ** 2) + 10e-20))
    return overall_snr
