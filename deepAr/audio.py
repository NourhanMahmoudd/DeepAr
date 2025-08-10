import torch
import torchaudio.transforms as T
from typing import Union
import numpy as np
from subprocess import CalledProcessError, run
SAMPLE_RATE = 16000

def resample_audio(audio: torch.Tensor, orig_sr: int, target_sr: int = 16000) -> torch.Tensor:
    """Resample audio tensor to target sampling rate"""
    if orig_sr != target_sr:
        resampler = T.Resample(orig_sr, target_sr)
        audio = resampler(audio)
    return audio

def load(
    file: Union[str, np.ndarray, torch.Tensor],

    target_sr: int = 16000
) -> torch.Tensor:
    """Load and preprocess audio file, ensuring it is resampled to target_sr (16kHz)"""
    
    speech = None

    if isinstance(file, str):
        # Load from file path
        speech = load_audio(file)

    elif isinstance(file , np.ndarray):
        speech = torch.from_numpy(file)

    elif isinstance(file, torch.Tensor):
        speech = file

    else:
        raise ValueError(f"Unsupported audio input type: {type(file)}")
    

    return  speech


def load_audio(file: str, sr: int = SAMPLE_RATE):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A torch.Tensor NumPy array containing the audio waveform, in float32 dtype.
    """
    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
    # fmt: off
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]
    # fmt: on
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return torch.Tensor(np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0)
