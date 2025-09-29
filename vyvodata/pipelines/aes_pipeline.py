from vyvodata.tools.audiobox_aesthetics.infer import initialize_predictor
from typing import Optional, Union, List, Dict, Any
import os
import json

# Import huggingface utils
from vyvodata.utils.huggingface import download_audio_files


class AudioAestheticsPredictor:
    def __init__(
        self,
        checkpoint_pth: Optional[str] = None,
    ):
        """
        Initialize the AudioAestheticsPredictor class with a predictor.
        """
        self.model = None
        if self.model is None:
            self.load_model(checkpoint_pth=checkpoint_pth)

    def load_model(self, checkpoint_pth: Optional[str] = None):
        """
        Load the model for the pipeline.
        """
        self.predictor = initialize_predictor(checkpoint_pth=checkpoint_pth)

    def process_audio(self, audio_paths):
        """
        Process audio files using the initialized predictor.

        Args:
            audio_paths (list): List of dictionaries containing paths to audio files.
                Each dictionary should have a 'path' key with the file path as value.

        Returns:
            The results from the predictor's forward method.
        """
        output = self.predictor.forward(audio_paths)
        return output

    def __call__(
        self, audio_path: Union[str, List[str]], output_dir: str = "./results", **kwargs
    ) -> Dict[str, Any]:
        """
        Make the class callable to directly process audio files or HuggingFace datasets.

        Args:
            audio_path: Either a single audio file path, list of audio file paths, or a HuggingFace dataset ID.
            output_dir: Directory to save results when processing datasets
            **kwargs: Additional arguments for dataset processing (split, num_samples, etc.)

        Returns:
            Dictionary containing the prediction results.

        Examples:
            >>> predictor = AudioAestheticsPredictor()

            # Process a single audio file
            >>> result = predictor("path/to/audio.wav")

            # Process multiple audio files
            >>> results = predictor(["path/to/audio1.wav", "path/to/audio2.wav"])

            # Process a HuggingFace dataset
            >>> scores = predictor("OpenSpeechHub/2M-Belebele-Ja", num_samples=10)
        """
        # Check if this is a HuggingFace dataset ID
        if (
            isinstance(audio_path, str)
            and not os.path.exists(audio_path)
            and "/" in audio_path
        ):
            # This looks like a HF dataset ID (e.g., "OpenSpeechHub/2M-Belebele-Ja")
            print(f"Processing HuggingFace dataset: {audio_path}")
            return self.process_hf_dataset_simple(
                dataset_name=audio_path, output_dir=output_dir, **kwargs
            )

        # Handle audio file paths
        if isinstance(audio_path, str):
            # Check if file exists
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            # Format single path as a list with one item
            formatted_paths = [{"path": audio_path}]
        else:
            # Format list of paths
            for path in audio_path:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Audio file not found: {path}")
            formatted_paths = [{"path": path} for path in audio_path]

        # Process the audio and return results
        result = self.process_audio(formatted_paths)

        # If single file, convert any tensor values to Python types
        if isinstance(audio_path, str):
            if isinstance(result, list) and len(result) == 1:
                result = result[0]

            # Clean result values for JSON compatibility
            if isinstance(result, dict):
                for key, value in result.items():
                    if hasattr(value, "item"):
                        result[key] = round(float(value.item()), 3)
                    elif isinstance(value, (int, float)):
                        result[key] = round(float(value), 3)

        return result

    def process_hf_dataset_simple(
        self,
        dataset_name: str,
        output_dir: str = "./downloaded_audio",
        split: str = "train",
        audio_column: str = "audio",
        num_samples: Optional[int] = None,
        id_column: Optional[str] = None,
        save_json: bool = True,
    ) -> Dict[str, Any]:
        """
        Simplified function to process audio from a HuggingFace dataset and return aesthetics scores.

        Args:
            dataset_name: Name of the HuggingFace dataset
            output_dir: Directory to save downloaded audio files and results
            split: Dataset split to use
            audio_column: Name of the column containing audio data
            num_samples: Number of samples to process (None for all)
            id_column: Column to use for audio file IDs (None for default)
            save_json: Whether to save results to a JSON file

        Returns:
            Dictionary with file-specific scores and average scores
        """
        # Create output subdirectory based on dataset name
        dataset_output_dir = os.path.join(
            output_dir, dataset_name.replace("/", "_") + f"_{split}"
        )
        os.makedirs(dataset_output_dir, exist_ok=True)

        # Download audio files from the dataset
        audio_files = download_audio_files(
            dataset_name=dataset_name,
            output_dir=dataset_output_dir,
            split=split,
            audio_column=audio_column,
            num_samples=num_samples,
            id_column=id_column,
        )

        # Check if we got any audio files
        if not audio_files:
            return {}

        # Run inference on all downloaded audio files
        results = self.process_audio([{"path": path} for path in audio_files])

        # Format results to include both file paths and scores
        file_scores = {}
        avg_scores = {}

        if results:
            # Get metrics from the first result
            metrics = list(results[0].keys())

            # Initialize averages
            for metric in metrics:
                avg_scores[metric] = 0

            # Process each file's results
            for i, path in enumerate(audio_files):
                # Get filename without directory
                filename = os.path.basename(path)

                # Store scores for this file
                file_scores[filename] = {}

                # Process each metric
                for metric in metrics:
                    value = results[i].get(metric)
                    if hasattr(value, "item"):
                        value = float(value.item())
                    if value is not None:
                        # Round to 3 decimal places
                        rounded_value = round(float(value), 3)
                        file_scores[filename][metric] = rounded_value
                        avg_scores[metric] += rounded_value

            # Calculate final averages
            if file_scores:
                for metric in metrics:
                    avg_scores[metric] = round(avg_scores[metric] / len(file_scores), 3)

        # Prepare final output
        output = {"files": file_scores, "average": avg_scores}

        # Save results as JSON
        if save_json and output["files"]:
            output_path = os.path.join(dataset_output_dir, "scores.json")
            with open(output_path, "w") as f:
                json.dump(output, f, indent=2)

        return output
