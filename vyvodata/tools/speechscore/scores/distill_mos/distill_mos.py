import numpy as np
import torch

from vyvodata.tools.speechscore.basis import ScoreBasis
from vyvodata.tools.speechscore.scores.distill_mos.sqa import ConvTransformerSQAModel


class DistillMos(ScoreBasis):
    def __init__(self):
        super().__init__(name="DISTILL_MOS")
        self.intrusive = False
        self.score_rate = 16000
        self.model = ConvTransformerSQAModel()
        self.model.eval()

    def windowed_scoring(self, audios, score_rate):
        score = self.model(torch.from_numpy(np.expand_dims(audios[0], axis=0)).float())
        score_np = score.detach().cpu().numpy()
        return score_np[0][0]
