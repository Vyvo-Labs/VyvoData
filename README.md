<div align="center">
<h2>
    VyvoData: Data Management and Preprocessing Tools for HF Datasets
</h2>
<img width="500" alt="teaser" src="assets/logo.png">
</div>

## 🛠️ Installation

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

## 🎙️ Usage

### Audio Quality Assessment

```python
from vyvodata.tools.audiobox_aesthetics.infer import audiobox_aesthetics_predict

results = audiobox_aesthetics_predict(
    input_file="path/to/audio/file.wav",
    ckpt="facebook/audiobox-aesthetics",
    batch_size=1,
)
```

### Speech Quality Assessment

```python
from vyvodata.pipelines.speechscore_pipeline import SpeechScorePredictor

predictor = SpeechScorePredictor(
    metrics=["DNSMOS"]  # STOI, PESQ, NISQA
)

scores = predictor(
    test_path="path/to/audio/file.wav",
    reference_path="path/to/reference/file.wav",
)
```

## 😍 Contributing

```bash
uv pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## 📜 License

This project is licensed under the terms of the Apache License 2.0.

## 🤗 Citation

```bibtex
@article{tjandra2025aes,
    title={Meta Audiobox Aesthetics: Unified Automatic Quality Assessment for Speech, Music, and Sound},
    author={Andros Tjandra and Yi-Chiao Wu and Baishan Guo and John Hoffman and Brian Ellis and Apoorv Vyas and Bowen Shi and Sanyuan Chen and Matt Le and Nick Zacharov and Carleigh Wood and Ann Lee and Wei-Ning Hsu},
    year={2025},
    url={https://arxiv.org/abs/2502.05139}
}
```
