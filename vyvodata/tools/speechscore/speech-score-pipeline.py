# Import pprint for pretty-printing the results in a more readable format
import pprint
# Import the SpeechScore class to evaluate speech quality metrics
from speechscore import SpeechScore 

# Main block to ensure the code runs only when executed directly
if __name__ == '__main__':
    # Initialize a SpeechScore object with a list of score metrics to be evaluated
    # Supports any subsets of the list
    
    # Non-intrusive tests ['NISQA', 'DNSMOS', 'DISTILL_MOS', SRMR'] : No reference audio is required
    
    mySpeechScore = SpeechScore([
        'SRMR', 'PESQ', 'NB_PESQ', 'STOI', 'SISDR', 
        'FWSEGSNR', 'LSD', 'BSSEval', 'DNSMOS', 
        'SNR', 'SSNR', 'LLR', 'CSIG', 'CBAK', 
        'COVL', 'MCD', 'NISQA', 'DISTILL_MOS'
    ])
    
    # Call the SpeechScore object to evaluate the speech metrics between 'noisy' and 'clean' audio
    # Arguments:
    # - {test_path, reference_path} supports audio directories or audio paths (.wav or .flac)
    # - window (float): seconds, set None to specify no windowing (process the full audio)
    # - score_rate (int): specifies the sampling rate at which the metrics should be computed
    # - return_mean (bool): set True to specify that the mean score for each metric should be returned

    
    print('score for a signle wav file')
    scores = mySpeechScore(test_path='audios/noisy.wav', reference_path='audios/clean.wav', window=None, score_rate=16000, return_mean=False)
    #scores = mySpeechScore(test_path='audios/noisy.wav', reference_path=None) # for Non-instrusive tests
    # Pretty-print the resulting scores in a readable format
    pprint.pprint(scores)

    print('score for wav directories')
    scores = mySpeechScore(test_path='audios/noisy/', reference_path='audios/clean/', window=None, score_rate=16000, return_mean=True)

    # Pretty-print the resulting scores in a readable format
    pprint.pprint(scores)

    # Print only the resulting mean scores in a readable format
    #pprint.pprint(scores['Mean_Score'])
