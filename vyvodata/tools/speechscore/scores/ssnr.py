from basis import ScoreBasis
import numpy as np

class SSNR(ScoreBasis):
    def __init__(self):
        super(SSNR, self).__init__(name='SSNR')
        self.intrusive = False

    def windowed_scoring(self, audios, score_rate):
        if len(audios) != 2:
            raise ValueError('SSNR needs a reference and a test signals.')
        return cal_SSNR(audios[0], audios[1], score_rate)

def cal_SSNR(ref_wav, deg_wav, srate=16000, eps=1e-10):
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
   
    # global variables
    winlength = int(np.round(30 * srate / 1000)) # 30 msecs
    skiprate  = winlength // 4
    MIN_SNR   = -10
    MAX_SNR   = 35

    # For each frame, calculate SSNR
    num_frames    = int(clean_length / skiprate - (winlength/skiprate))
    start         = 0
    time          = np.linspace(1, winlength, winlength) / (winlength + 1)
    window        = 0.5 * (1 - np.cos(2 * np.pi * time))
    segmental_snr = []

    for frame_count in range(int(num_frames)):
        # (1) get the frames for the test and ref speech.
        # Apply Hanning Window
        clean_frame     = clean_speech[start:start+winlength]
        processed_frame = processed_speech[start:start+winlength]
        clean_frame     = clean_frame * window
        processed_frame = processed_frame * window

        # (2) Compute Segmental SNR
        signal_energy = np.sum(clean_frame ** 2)
        noise_energy  = np.sum((clean_frame - processed_frame) ** 2)
        segmental_snr.append(10 * np.log10(signal_energy / (noise_energy + eps)+ eps))
        segmental_snr[-1] = max(segmental_snr[-1], MIN_SNR)
        segmental_snr[-1] = min(segmental_snr[-1], MAX_SNR)
        start += int(skiprate)
    return sum(segmental_snr) / len(segmental_snr)
