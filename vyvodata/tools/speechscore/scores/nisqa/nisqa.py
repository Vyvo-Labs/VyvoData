import os

from vyvodata.tools.speechscore.basis import ScoreBasis
from vyvodata.tools.speechscore.scores.nisqa.cal_nisqa import load_nisqa_model


class NISQA(ScoreBasis):
    def __init__(self):
        super().__init__(name="NISQA")
        self.intrusive = False
        self.score_rate = 48000
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "weights", "nisqa.tar")
        self.model = load_nisqa_model(model_path, device="cpu")

    def windowed_scoring(self, audios, score_rate):
        from vyvodata.tools.speechscore.scores.nisqa.cal_nisqa import cal_nisqa

        score = cal_nisqa(self.model, audios[0])
        return score
