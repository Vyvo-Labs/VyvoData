from basis import ScoreBasis

class PESQ(ScoreBasis):
    def __init__(self):
        super(PESQ, self).__init__(name='PESQ')
        self.intrusive = False
        self.mono = True
        self.score_rate = 16000

    def windowed_scoring(self, audios, rate):
        from pesq import pesq
        if len(audios) != 2:
            raise ValueError('PESQ needs a reference and a test signals.')
            return None
        return pesq(rate, audios[1], audios[0], 'wb')

