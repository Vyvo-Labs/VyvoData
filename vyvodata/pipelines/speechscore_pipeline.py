from vyvodata.tools.speechscore import SpeechScore
from typing import Optional, Union, List, Dict, Any
import os
import json


class SpeechScorePredictor:
    def __init__(
        self,
        metrics: Optional[List[str]] = None,
    ):
        """
        Initialize the SpeechScorePredictor class with speech quality metrics.

        Args:
            metrics: List of metrics to evaluate. If None, uses all available metrics.
                Available metrics: 'SRMR', 'PESQ', 'NB_PESQ', 'STOI', 'SISDR',
                'FWSEGSNR', 'LSD', 'BSSEval', 'DNSMOS', 'SNR', 'SSNR', 'LLR',
                'CSIG', 'CBAK', 'COVL', 'MCD', 'NISQA', 'DISTILL_MOS'

        Note:
            Non-intrusive metrics (no reference needed): 'NISQA', 'DNSMOS', 'DISTILL_MOS', 'SRMR'
        """
        self.model = None
        self.metrics = metrics
        if self.model is None:
            self.load_model(metrics=metrics)

    def load_model(self, metrics: Optional[List[str]] = None):
        """
        Load the SpeechScore model with specified metrics.

        Args:
            metrics: List of metrics to evaluate. If None, uses all available metrics.
        """
        if metrics is None:
            # Use all available metrics by default
            metrics = [
                'SRMR', 'PESQ', 'NB_PESQ', 'STOI', 'SISDR',
                'FWSEGSNR', 'LSD', 'BSSEval', 'DNSMOS',
                'SNR', 'SSNR', 'LLR', 'CSIG', 'CBAK',
                'COVL', 'MCD', 'NISQA', 'DISTILL_MOS'
            ]

        self.model = SpeechScore(metrics)

    def process_audio(
        self,
        test_path: str,
        reference_path: Optional[str] = None,
        window: Optional[float] = None,
        score_rate: int = 16000,
        return_mean: bool = False
    ):
        """
        Process audio files using the initialized SpeechScore model.

        Args:
            test_path: Path to test audio file or directory (.wav or .flac)
            reference_path: Path to reference audio file or directory. Optional for non-intrusive metrics.
            window: Window size in seconds. None for full audio processing.
            score_rate: Sampling rate for metric computation.
            return_mean: Whether to return mean scores for directory processing.

        Returns:
            Dictionary containing speech quality scores.
        """
        scores = self.model(
            test_path=test_path,
            reference_path=reference_path,
            window=window,
            score_rate=score_rate,
            return_mean=return_mean
        )
        return scores

    def __call__(
        self,
        test_path: Union[str, List[str]],
        reference_path: Optional[Union[str, List[str]]] = None,
        window: Optional[float] = None,
        score_rate: int = 16000,
        return_mean: bool = False,
        output_dir: Optional[str] = None,
        save_json: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make the class callable to directly process audio files.

        Args:
            test_path: Path(s) to test audio file(s) or directory (.wav or .flac)
            reference_path: Path(s) to reference audio file(s) or directory. Optional for non-intrusive metrics.
            window: Window size in seconds. None for full audio processing.
            score_rate: Sampling rate for metric computation (default: 16000).
            return_mean: Whether to return mean scores for directory processing.
            output_dir: Directory to save results (optional).
            save_json: Whether to save results to a JSON file.
            **kwargs: Additional arguments for future extensions.

        Returns:
            Dictionary containing the prediction results.

        Examples:
            >>> predictor = SpeechScorePredictor()

            # Process a single audio file (with reference)
            >>> result = predictor("test.wav", reference_path="clean.wav")

            # Process without reference (non-intrusive metrics only)
            >>> result = predictor("test.wav")

            # Process multiple files from directories
            >>> results = predictor("noisy_dir/", reference_path="clean_dir/", return_mean=True)

            # Process with custom metrics
            >>> predictor = SpeechScorePredictor(metrics=['PESQ', 'STOI', 'SISDR'])
            >>> result = predictor("test.wav", reference_path="clean.wav")
        """
        # Handle single file or directory
        if isinstance(test_path, str):
            if not os.path.exists(test_path):
                raise FileNotFoundError(f"Test audio file/directory not found: {test_path}")

            # Check reference path if provided
            if reference_path is not None and not os.path.exists(reference_path):
                raise FileNotFoundError(f"Reference audio file/directory not found: {reference_path}")

            # Process the audio
            result = self.process_audio(
                test_path=test_path,
                reference_path=reference_path,
                window=window,
                score_rate=score_rate,
                return_mean=return_mean
            )

        else:
            # Handle list of paths (future extension)
            raise NotImplementedError("List of audio paths not yet supported. Use directory path instead.")

        # Clean result values for JSON compatibility
        result = self._clean_result(result)

        # Save to JSON if requested
        if save_json and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "speechscore_results.json")
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to: {output_path}")

        return result

    def _clean_result(self, result: Any) -> Any:
        """
        Clean result values for JSON compatibility.

        Args:
            result: Raw result from SpeechScore model.

        Returns:
            Cleaned result with JSON-compatible types.
        """
        if isinstance(result, dict):
            cleaned = {}
            for key, value in result.items():
                if isinstance(value, dict):
                    cleaned[key] = self._clean_result(value)
                elif hasattr(value, "item"):
                    cleaned[key] = round(float(value.item()), 4)
                elif isinstance(value, (int, float)):
                    cleaned[key] = round(float(value), 4)
                else:
                    cleaned[key] = value
            return cleaned
        elif isinstance(result, list):
            return [self._clean_result(item) for item in result]
        else:
            return result
