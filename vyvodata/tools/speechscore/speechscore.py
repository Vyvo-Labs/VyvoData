import os
import librosa
import soundfile as sf
import resampy
import numpy as np
from scores.srmr.srmr import SRMR
from scores.dnsmos.dnsmos import DNSMOS
from scores.pesq import PESQ
from scores.nb_pesq import NB_PESQ
from scores.sisdr import SISDR
from scores.stoi import STOI
from scores.fwsegsnr import FWSEGSNR
from scores.lsd import LSD
from scores.bsseval import BSSEval
from scores.snr import SNR
from scores.ssnr import SSNR
from scores.llr import LLR
from scores.csig import CSIG
from scores.cbak import CBAK
from scores.covl import COVL
from scores.mcd import MCD
from scores.nisqa.nisqa import NISQA
from scores.distill_mos.distill_mos import DISTILL_MOS 

def compute_mean_results(*results):
    mean_result = {}

    # Use the first dictionary as a reference for keys
    for key in results[0]:
        # If the value is a nested dictionary, recurse
        if isinstance(results[0][key], dict):
            nested_results = [d[key] for d in results]
            mean_result[key] = compute_mean_results(*nested_results)
        # Otherwise, compute the mean of the values
        else:
            mean_result[key] = sum(d[key] for d in results) / len(results)

    return mean_result

class ScoresList:
    def __init__(self):
        self.scores = []

    def __add__(self, score):
        self.scores += [score]
        return self

    def __str__(self):
        return 'Scores: ' + ' '.join([x.name for x in self.scores])

    def __call__(self, test_path, reference_path, window=None, score_rate=None, return_mean=False):
        """
        window: float
            the window length in seconds to use for scoring the files.
        score_rate:
            the sampling rate specified for scoring the files.
        """
        if test_path is None:
            print(f'Please provide audio path for test_path')
            return
        results = {}
             
        if os.path.isdir(test_path):
            audio_list = self.get_audio_list(test_path)
            if audio_list is None: return
            for audio_id in audio_list:
                results_id = {}                
                if reference_path is not None:
                    data = self.audio_reader(test_path+'/'+audio_id, reference_path+'/'+audio_id)
                else:
                    data = self.audio_reader(test_path+'/'+audio_id, None)
                for score in self.scores:
                    result_score = score.scoring(data, window, score_rate)
                    results_id[score.name] = result_score
                results[audio_id] = results_id
        else:            
            data = self.audio_reader(test_path, reference_path)
            for score in self.scores:
                result_score = score.scoring(data, window, score_rate)
                results[score.name] = result_score

        if return_mean:
            mean_result = compute_mean_results(*results.values())
            results['Mean_Score'] = mean_result

        return results

    def get_audio_list(self, path):
        # Initialize an empty list to store audio file names
        audio_list = []

        # Find all '.wav' audio files in the given path
        path_list = librosa.util.find_files(path, ext="wav")

        # If no '.wav' files are found, try to find '.flac' audio files instead
        if len(path_list) == 0:
            path_list = librosa.util.find_files(path, ext="flac")

        # If no audio files are found at all, print an error message and return None
        if len(path_list) == 0:
            print(f'No audio files found in {path}, scoring ended!')
            return None

        # Loop through the list of found audio file paths
        for audio_path in path_list:
            # Split the file path by '/' and append the last element (the file name) to the audio_list
            audio_path_s = audio_path.split('/')
            audio_list.append(audio_path_s[-1])

        # Return the list of audio file names
        return audio_list

    def audio_reader(self, test_path, reference_path):
        """loading sound files and making sure they all have the same lengths
            (zero-padding to the largest). Also works with numpy arrays.
        """
        data = {}
        audios = []
        maxlen = 0
        audio_test, rate_test = sf.read(test_path, always_2d=True)

        if audio_test.shape[1] > 1:
            audio_test = audio_test[..., 0, None]

        rate = rate_test
        if reference_path is not None:
            audio_ref, rate_ref = sf.read(reference_path, always_2d=True)
            if audio_ref.shape[1] > 1:
                audio_ref = audio_ref[..., 0, None]
            if rate_test != rate_ref:
                rate = min(rate_test, rate_ref)
            if rate_test != rate:
                audio_test = resampy.resample(audio_test, rate_test, rate, axis=0)
            if rate_ref != rate:
                audio_ref = resampy.resample(audio_ref, rate_ref, rate, axis=0)
            audios += [audio_test]
            audios += [audio_ref]
        else:
            audios += [audio_test]

        maxlen = 0
        for index, audio in enumerate(audios):
            maxlen = max(maxlen, audio.shape[0])
        ##padding
        for index, audio in enumerate(audios):
            if audio.shape[0] != maxlen:
                new = np.zeros((maxlen,))
                new[:audio.shape[0]] = audio[...,0]
                audios[index] = new
            else:
                audios[index] = audio[...,0]
        data['audio'] = audios
        data['rate'] = rate
        return data

def SpeechScore(scores=''):
    """ Load the desired scores inside a Metrics object that can then
    be called to compute all the desired scores.

    Parameters:
    ----------
    scores: str or list of str
        the scores matching any of these will be automatically loaded. this
        match is relative to the structure of the speechscores package.
        For instance:
        * 'absolute' will match all non-instrusive scores
        * 'absolute.srmr' or 'srmr' will only match SRMR
        * '' will match all

    Returns:
    --------

    A ScoresList object, that can be run to get the desired scores
    """

    score_cls = ScoresList()
    for score in scores:
        if score.lower() == 'srmr':
            score_cls += SRMR()
        elif score.lower() == 'pesq':
            score_cls += PESQ()
        elif score.lower() == 'nb_pesq':
            score_cls += NB_PESQ()
        elif score.lower() == 'stoi':
            score_cls += STOI()
        elif score.lower() == 'sisdr':
            score_cls += SISDR()
        elif score.lower() == 'fwsegsnr':
            score_cls += FWSEGSNR()
        elif score.lower() == 'lsd':
            score_cls += LSD()
        elif score.lower() == 'bsseval':
            score_cls += BSSEval()
        elif score.lower() == 'dnsmos':
            score_cls += DNSMOS()
        elif score.lower() == 'snr':
            score_cls += SNR()
        elif score.lower() == 'ssnr':
            score_cls += SSNR()
        elif score.lower() == 'llr':
            score_cls += LLR()
        elif score.lower() == 'csig':
            score_cls += CSIG()
        elif score.lower() == 'cbak':
            score_cls += CBAK()
        elif score.lower() == 'covl':
            score_cls += COVL()
        elif score.lower() == 'mcd':
            score_cls += MCD()
        elif score.lower() == 'nisqa':
            score_cls += NISQA()
        elif score.lower() == 'distill_mos':
        	score_cls += DISTILL_MOS()
        else:
           print('score is pending implementation...')
    return score_cls
