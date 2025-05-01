import os
import torch
import warnings
from typing import Optional, Union, List
from pathlib import Path

def get_device(device: Optional[str] = None) -> str:
    """
    Get the appropriate device for model inference.
    
    Args:
        device: Manually specified device (optional)
    Returns:
        str: 'cuda' if GPU is available, 'cpu' otherwise
    """
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"

def validate_audio_file(file_path: Union[str, Path]) -> bool:
    """
    Validate if the audio file exists and has a supported format.
    
    Args:
        file_path: Path to the audio file
    Returns:
        bool: True if valid, False otherwise
    """
    supported_formats = {'.wav', '.mp3', '.m4a', '.flac'}
    path = Path(file_path)
    
    if not path.exists():
        warnings.warn(f"File not found: {file_path}")
        return False
        
    if path.suffix.lower() not in supported_formats:
        warnings.warn(f"Unsupported audio format: {path.suffix}")
        return False
        
    return True

def format_timestamp(seconds: float) -> str:
    """
    Convert seconds to formatted timestamp (HH:MM:SS.mmm).
    
    Args:
        seconds: Time in seconds
    Returns:
        str: Formatted timestamp
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

def batch_process_directory(
    directory: Union[str, Path],
    valid_extensions: Optional[List[str]] = None
) -> List[Path]:
    """
    Get all audio files from a directory.
    
    Args:
        directory: Directory path
        valid_extensions: List of valid file extensions (default: ['.wav', '.mp3', '.m4a', '.flac'])
    Returns:
        List[Path]: List of valid audio file paths
    """
    if valid_extensions is None:
        valid_extensions = ['.wav', '.mp3', '.m4a', '.flac']
        
    directory = Path(directory)
    audio_files = []
    
    for ext in valid_extensions:
        audio_files.extend(directory.glob(f"*{ext}"))
    
    return sorted(audio_files)

def ensure_output_dir(output_path: Union[str, Path]) -> Path:
    """
    Ensure output directory exists, create if necessary.
    
    Args:
        output_path: Directory path
    Returns:
        Path: Path object of the output directory
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path