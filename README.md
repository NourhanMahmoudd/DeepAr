# DeepAr Model Wrapper

A simple Python package that wraps the [CUAIStudents/DeepAr](https://huggingface.co/CUAIStudents/DeepAr) model for **Arabic speech recognition (ASR)**.  
It provides a clean interface for transcribing single audio files or batches with optional timestamps.

> ⚠️ This repository is an **inference wrapper only**. Training code is not included.
> 
## Installation

1. Clone the repository:
```bash
git clone https://github.com/NourhanMahmoudd/DeepAr.git
cd DeepAr
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Make sure **FFmpeg** is installed and available in your path.

## Quickstart

```python
from deepar import DeepAr

# Initialize the model
model = DeepAr()

# Transcribe an audio file
text = model.transcribe("audio.wav")
print(text)

# Get timestamps
result = model.transcribe("audio.wav", return_timestamps=True)

# Batch processing
audios = ["audio1.wav", "audio2.wav"]
results = model.transcribe_batch(audios)
print(results)
```

## Features
- Simple Python interface for the Hugging Face DeepAr model
- Single-file and batch transcription
- Word- and segment-level timestamps (optional)
- Handles multiple input formats (.wav, .mp3, .flac, raw arrays, tensors)
- Automatic resampling to 16kHz for compatibility
- GPU support (CUDA) when available

## API

### DeepAr

```python
DeepAr(
    model_name: str = "CUAIStudents/DeepAr",
    device: Optional[str] = None,  # "cpu" or "cuda"
    chunk_length_s: int = 30,      # Process audio in chunks (seconds)
    stride_length_s: List[int] = [5, 5]  # Overlap between chunks
)
```

### transcribe

```python
transcribe(
    audio: Union[str, bytes, torch.Tensor, numpy.ndarray],
    sample_rate: Optional[int] = None,
    return_timestamps: Union[bool, str] = False,
    **generation_kwargs
)
```

### transcribe_batch

```python
transcribe_batch(
    audios: List[Union[str, bytes, torch.Tensor, numpy.ndarray]],
    return_timestamps: Union[bool, str] = True
) 
```

