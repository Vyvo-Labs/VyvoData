"""
Utility functions for working with Hugging Face datasets.
Includes functions for downloading audio files and specific datasets like Emilia.
"""

import os
from typing import List, Optional
from tqdm.auto import tqdm
import soundfile as sf
from datasets import load_dataset, Audio


def download_hf(
    repo_id="kadirnar/test",
    repo_type="model",
    ignore_patterns=["*.md", "*..gitattributes"],
    local_dir=None,
    allow_patterns=None,
):
    """
    Downloads a model from Hugging Face Hub.

    Args:
        repo_id (str): The repository ID on Hugging Face Hub.
        repo_type (str): Type of repository ('model', 'dataset', etc.).
        ignore_patterns (list): Patterns to ignore during download.
        local_dir (str): Local directory to save the model. Defaults to repo_id's last component.
        allow_patterns (str): Patterns to allow during download.

    Returns:
        str: Path to the downloaded model.
    """
    from huggingface_hub import snapshot_download

    if local_dir is None:
        local_dir = repo_id.split("/")[-1]

    return snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        ignore_patterns=ignore_patterns,
        local_dir=local_dir,
        allow_patterns=allow_patterns,
    )


def upload_to_hub(
    local_dir,
    repo_id,
    repo_type="model",
    commit_message=None,
    private=False,
    token=None,
    create_repo=True,
    ignore_patterns=None,
):
    """
    Uploads local content to Hugging Face Hub.

    Args:
        local_dir (str): Path to the local directory to upload.
        repo_id (str): The repository ID on Hugging Face Hub.
        repo_type (str): Type of repository ('model', 'dataset', etc.).
        commit_message (str): Commit message for the upload. Defaults to 'Upload {repo_type}'.
        private (bool): Whether the repository should be private.
        token (str): HuggingFace token. Will use the cached token if not provided.
        create_repo (bool): Whether to create the repository if it doesn't exist.
        ignore_patterns (list): Patterns to ignore during upload.

    Returns:
        str: URL of the repository on Hugging Face Hub.
    """
    from huggingface_hub import HfApi

    if commit_message is None:
        commit_message = f"Upload {repo_type}"

    if ignore_patterns is None:
        ignore_patterns = [".git/**", ".gitignore", "**/.DS_Store", "**/__pycache__/**"]

    api = HfApi(token=token)

    # Create repository if needed
    if create_repo:
        api.create_repo(
            repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True
        )

    # Upload directory content to the Hub
    url = api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=commit_message,
        ignore_patterns=ignore_patterns,
    )

    return url


def upload_large_files_to_hub(
    file_paths,
    repo_id,
    repo_type="model",
    path_in_repo=None,
    commit_message=None,
    private=False,
    token=None,
    create_repo=True,
    max_shard_size="500MB",
):
    """
    Uploads large files to Hugging Face Hub with chunking support.

    Args:
        file_paths (str or list): Path(s) to the large file(s) to upload.
        repo_id (str): The repository ID on Hugging Face Hub.
        repo_type (str): Type of repository ('model', 'dataset', etc.).
        path_in_repo (str): Path in the repository where the file(s) should be stored.
                           If None, files will be stored at the root.
        commit_message (str): Commit message for the upload. Defaults to 'Upload large files'.
        private (bool): Whether the repository should be private.
        token (str): HuggingFace token. Will use the cached token if not provided.
        create_repo (bool): Whether to create the repository if it doesn't exist.
        max_shard_size (str): Maximum size for the chunks in bytes (e.g. "500MB", "1GB").

    Returns:
        str: URL of the repository on Hugging Face Hub.
    """
    import os
    from huggingface_hub import HfApi, upload_file

    if commit_message is None:
        commit_message = "Upload large files"

    if isinstance(file_paths, str):
        file_paths = [file_paths]

    api = HfApi(token=token)

    # Create repository if needed
    if create_repo:
        api.create_repo(
            repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True
        )

    uploaded_files = []

    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        destination_path = (
            file_name if path_in_repo is None else os.path.join(path_in_repo, file_name)
        )

        # Upload large file with potential chunking
        url = upload_file(
            path_or_fileobj=file_path,
            path_in_repo=destination_path,
            repo_id=repo_id,
            repo_type=repo_type,
            token=token,
            commit_message=f"{commit_message}: {file_name}",
            max_shard_size=max_shard_size,
        )

        uploaded_files.append(url)

    # Return the repository URL (common to all files)
    repo_url = f"https://huggingface.co/{repo_id}"
    return repo_url


def download_audio_files(
    dataset_name: str,
    output_dir: str,
    split: str = "train",
    audio_column: str = "audio",
    num_samples: Optional[int] = None,
    id_column: Optional[str] = None,
) -> List[str]:
    """
    Download audio files from a Hugging Face dataset and save them as WAV files.

    Args:
        dataset_name: Name of the dataset on Hugging Face (e.g., 'OpenSpeechHubCAVA/2M-Belebele-Ja')
        output_dir: Directory to save the downloaded audio files
        split: Dataset split to download (e.g., 'train', 'validation', 'test')
        audio_column: Name of the column containing audio data
        num_samples: If provided, only download this many samples (useful for testing)
        id_column: Column to use as the filename prefix. If None, will use index numbers.

    Returns:
        List of paths to the saved audio files
    """
    # Simple implementation

    print(f"Loading dataset: {dataset_name}, split: {split}")

    try:
        # Load dataset
        dataset = load_dataset(dataset_name, split=split)

        # Take a sample if requested
        if num_samples is not None and num_samples < len(dataset):
            dataset = dataset.shuffle(seed=42).select(range(num_samples))
            print(f"Selected {num_samples} samples from the dataset")

        # Ensure dataset has audio column
        if audio_column not in dataset.column_names:
            raise ValueError(
                f"Audio column '{audio_column}' not found in dataset. "
                f"Available columns: {dataset.column_names}"
            )

        # Cast to Audio feature if needed
        if not isinstance(dataset.features[audio_column], Audio):
            print(f"Converting {audio_column} column to Audio feature...")
            dataset = dataset.cast_column(audio_column, Audio())

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Download and save audio files
        saved_files = []
        for i, item in enumerate(tqdm(dataset, desc="Downloading audio files")):
            try:
                # Get audio data
                audio_data = item[audio_column]

                # Generate file name
                if id_column and id_column in item:
                    # Use the provided column as file name prefix
                    file_id = str(item[id_column])
                    # Clean up the ID to make a valid filename
                    file_id = "".join(c if c.isalnum() else "_" for c in file_id)
                else:
                    # Use index as file name
                    file_id = f"audio_{i:05d}"

                file_path = os.path.join(output_dir, f"{file_id}.wav")

                # Save audio file
                if (
                    isinstance(audio_data, dict)
                    and "array" in audio_data
                    and "sampling_rate" in audio_data
                ):
                    # Handle array format (most common from datasets)
                    sf.write(
                        file_path, audio_data["array"], audio_data["sampling_rate"]
                    )
                    saved_files.append(file_path)
                elif isinstance(audio_data, dict) and "path" in audio_data:
                    # Handle path format (copy file)
                    import shutil

                    shutil.copy(audio_data["path"], file_path)
                    saved_files.append(file_path)
                else:
                    print(f"Unsupported audio format for item {i}, skipping")

            except Exception as e:
                print(f"Error processing audio item {i}: {str(e)}")

        # No metadata file needed

        print(f"\nDownloaded {len(saved_files)} audio files to {output_dir}")
        return saved_files

    except Exception as e:
        print(f"Error downloading dataset {dataset_name}: {str(e)}")
        return []
