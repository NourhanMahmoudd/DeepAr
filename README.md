# DeepAr Model Wrapper

A simple Python wrapper for the [CUAIStudents/DeepAr](https://huggingface.co/CUAIStudents/DeepAr) Arabic speech recognition model from Hugging Face.

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

3. Make sure you have FFmpeg installed for audio processing.

## Usage

```python
from deepar import DeepAr

# Initialize the model
model = DeepAr()

# Transcribe an audio file
text = model.transcribe("audio.wav")
print(text)

# Get word-level timestamps
result = model.transcribe("audio.wav", return_timestamps=True)

# Process multiple files
audios = ["audio1.wav", "audio2.wav"]
transcriptions = model.transcribe_batch(audios)
```

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
    return_timestamps: bool = False,
    **generation_kwargs
)
```

### transcribe_batch

```python
transcribe_batch(
    audios: List[Union[str, bytes, torch.Tensor, numpy.ndarray]],
    return_timestamps: bool = True
) 
```

