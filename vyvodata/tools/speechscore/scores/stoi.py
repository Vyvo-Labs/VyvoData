from basis import ScoreBasis


class STOI(ScoreBasis):
    def __init__(self):
        super(STOI, self).__init__(name='STOI')
        self.intrusive = False
        self.mono = True

    def windowed_scoring(self, audios, score_rate):
        from pystoi.stoi import stoi
        if len(audios) != 2:
            raise ValueError('STOI needs a reference and a test signals.')

        return stoi(audios[1], audios[0], score_rate, extended=False)

