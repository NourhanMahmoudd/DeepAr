from typing import Optional, Union, List, Dict, Any
import torch
import torchaudio.functional as F
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from .utils import get_device, validate_audio_file
import numpy
import warnings

SAMPLE_RATE = 16000

class DeepAr:
    def __init__(
        self,
        model_name: str = "CUAIStudents/DeepAr",
        device: Optional[str] = None,
        chunk_length_s: int = 30,  # Add default chunking parameters
        stride_length_s: List[int] = [5, 5]
    ):
        self.device = get_device() if device is None else device
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,  
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True
        )
        self.model.to(self.device)
        
        # Make sure to use the processor that matches your fine-tuned model
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
            chunk_length_s=chunk_length_s,  # Process in chunks
            stride_length_s=stride_length_s  # Overlap between chunks
        )
    
    def transcribe(
        self,
        audio: Union[str, bytes, torch.Tensor, numpy.ndarray],
        sample_rate: Optional[int] = None,
        return_timestamps: Union[bool, str] = False,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[str, List[dict]]:
        """
        Transcribe audio using the Arabic Whisper model.
        
        Args:
            audio: Audio file path, bytes, or tensor
            sample_rate: Original sample rate of the audio if providing raw array/tensor
            return_timestamps: Whether to return word or segment timestamps
            generation_kwargs: Additional kwargs to pass to the generate method
            
        Returns:
            Transcription text or dictionary with timestamps
        """
        # Handle tensor/array inputs
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()
        if isinstance(audio, numpy.ndarray):
            if sample_rate is None:
                warnings.warn("Expected sample rate: 16K Hz. Please ensure the audio is correctly sampled, or specify the original sample rate using sample_rate=.")
            else:
                audio = torch.from_numpy(audio)
                audio = F.resample(audio, sample_rate, SAMPLE_RATE)
                audio = audio.numpy()
                
        # Validate file path
        if isinstance(audio, str):
            if not validate_audio_file(audio):
                raise ValueError("Invalid audio file, file path does not exist")
        
        transcription = self.pipe(
            audio, 
            return_timestamps=return_timestamps, 
            generate_kwargs=generation_kwargs
        )
        
        # Return the full transcription object when timestamps are requested
        if return_timestamps:
            return transcription
        
        # Otherwise just return the text
        return transcription["text"]
        

