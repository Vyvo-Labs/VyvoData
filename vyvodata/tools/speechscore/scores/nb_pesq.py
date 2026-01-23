from vyvodata.tools.speechscore.basis import ScoreBasis


class NbPesq(ScoreBasis):
    def __init__(self):
        super().__init__(name="NB_PESQ")
        self.intrusive = False
        self.score_rate = 16000

    def windowed_scoring(self, audios, score_rate):
        from pesq import pesq

        if len(audios) != 2:
            raise ValueError("NB_PESQ needs a reference and a test signals.")
            return None
        return pesq(score_rate, audios[1], audios[0], "nb")
