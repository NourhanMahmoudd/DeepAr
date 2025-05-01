from typing import Optional, Union, List, Dict, Any
import torch
import torchaudio.functional as F
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from .utils import get_device, validate_audio_file
from huggingface_hub import login
import numpy as np
import warnings
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

class ArabicWhisper:
    def __init__(
        self,
        model_name: str = "CUAIStudents/DeepAr",
        device: Optional[str] = None,
        hf_token: Optional[str] = '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'
    ):
        try:
            self.device = get_device() if device is None else device
            self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            if hf_token:
                login(hf_token)
            elif os.environ.get("HF_TOKEN"):
                login(os.environ.get("HF_TOKEN"))
            else:
                logger.warning("No Hugging Face token provided. Some models may not be accessible.")
            
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_name,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True
            )
            self.model.to(self.device)
            
            # Use Whisper's processor since the fine-tuned model doesn't include one
            self.processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
            
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                torch_dtype=self.torch_dtype,
                device=self.device,
                chunk_length_s=30,
                stride_length_s=5,
            )
            logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise RuntimeError(f"Failed to initialize ArabicWhisper: {str(e)}")
    
    def _preprocess_audio(self, audio, sample_rate=None):
        """Preprocess audio to the correct format and sample rate."""
        try:
            if isinstance(audio, torch.Tensor):
                audio = audio.numpy()
            
            if isinstance(audio, np.ndarray):
                if sample_rate is None:
                    warnings.warn("Expected sample rate: 16K Hz. Please ensure the audio is correctly sampled, or specify the original sample rate using sample_rate=.")
                elif sample_rate != SAMPLE_RATE:
                    audio = torch.from_numpy(audio)
                    audio = F.resample(audio, sample_rate, SAMPLE_RATE)
                    audio = audio.numpy()
            
            elif isinstance(audio, str):
                if not validate_audio_file(audio):
                    raise ValueError(f"Invalid audio file: {audio}, file path does not exist")
            
            return audio
        except Exception as e:
            logger.error(f"Error preprocessing audio: {str(e)}")
            raise ValueError(f"Failed to preprocess audio: {str(e)}")

    def _generate_word_timestamps(self, text, chunks=None, total_duration=60.0):
        """Generate word-level timestamps from text and optional chunk timestamps."""
        words = text.split()
        word_list = []
        
        if not words:
            return word_list
        
        valid_chunks = []
        if chunks:
            valid_chunks = [c for c in chunks if c.get('timestamp') and 
                           (c['timestamp'][0] != 0.0 or c['timestamp'][1] != 0.0)]
        
        if valid_chunks:
            for chunk in valid_chunks:
                chunk_text = chunk['text'].strip()
                chunk_words = chunk_text.split()
                chunk_start, chunk_end = chunk['timestamp']
                
                if len(chunk_words) > 0:
                    word_duration = (chunk_end - chunk_start) / len(chunk_words)
                    
                    for i, word in enumerate(chunk_words):
                        word_start = chunk_start + i * word_duration
                        word_end = word_start + word_duration
                        
                        word_list.append({
                            'word': word,
                            'start': word_start,
                            'end': word_end
                        })
        else:
            word_duration = total_duration / max(len(words), 1)
            
            for i, word in enumerate(words):
                word_list.append({
                    'word': word,
                    'start': i * word_duration,
                    'end': (i + 1) * word_duration
                })
                
        return word_list

    def _fix_timestamps(self, result, return_timestamps):
        """Fix or add timestamps to the result."""
        if not return_timestamps:
            return result
            
        if 'chunks' in result:
            for chunk in result['chunks']:
                if chunk.get('timestamp') is None:
                    chunk['timestamp'] = [0.0, 0.0]
        
        if return_timestamps == "word" and 'words' not in result:
            logger.warning("Word-level timestamps requested but not returned by model. Creating approximate word timestamps.")
            result['words'] = self._generate_word_timestamps(
                result['text'], 
                result.get('chunks', [])
            )

        return result

    def _create_dummy_result(self, text, return_timestamps, total_duration=60.0):
        """Create a dummy result with timestamps when needed."""
        if not return_timestamps:
            return text
            
        if return_timestamps == "word":
            word_list = self._generate_word_timestamps(text, total_duration=total_duration)
            
            return {
                "text": text,
                "chunks": [{"text": text, "timestamp": [0.0, total_duration]}],
                "words": word_list
            }
        else:
            return {
                "text": text,
                "chunks": [{"text": text, "timestamp": [0.0, total_duration]}]
            }

    def transcribe(
        self,
        audio: Union[str, bytes, torch.Tensor, np.ndarray, List[Union[str, bytes, torch.Tensor, np.ndarray]]],
        sample_rate: Optional[int] = None,
        return_timestamps: Union[bool, str] = False,
        batch_size: int = 8,
        **kwargs
    ):
        try:
            is_batch = isinstance(audio, list)
            
            default_generate_kwargs = {
                "language": "ar",
                "task": "transcribe",
                "temperature": 0.2,
                "no_repeat_ngram_size": 3,
                "repetition_penalty": 1.5,
                "max_new_tokens": 225,  # Reduced from 448 to stay within limits
                "do_sample": True,
                "top_k": 50,
                "top_p": 0.95,
                "use_cache": True,
                "condition_on_prev_tokens": True,
                "max_length": 448  # Added explicit max_length
            }
            generate_kwargs = {**default_generate_kwargs, **kwargs}

            if "attn_implementation" in generate_kwargs:
                del generate_kwargs["attn_implementation"]
            
            if is_batch:
                if len(audio) > batch_size:
                    logger.info(f"Processing large batch of {len(audio)} items in chunks of {batch_size}")
                    all_results = []
                    for i in range(0, len(audio), batch_size):
                        batch_chunk = audio[i:i+batch_size]
                        logger.info(f"Processing batch chunk {i//batch_size + 1}/{(len(audio) + batch_size - 1)//batch_size}")

                        chunk_results = self.transcribe(
                            batch_chunk, 
                            sample_rate=sample_rate,
                            return_timestamps=return_timestamps,
                            batch_size=batch_size,
                            **kwargs
                        )
                        all_results.extend(chunk_results if isinstance(chunk_results, list) else [chunk_results])
                    
                    return all_results
                
                processed_audios = []
                for audio_item in audio:
                    processed_audios.append(self._preprocess_audio(audio_item, sample_rate))

                pipeline_kwargs = {
                    "return_timestamps": return_timestamps,
                    "generate_kwargs": generate_kwargs
                }
                
                if len(processed_audios) > 1:
                    if return_timestamps == "word":
                        adjusted_batch_size = min(2, batch_size, len(processed_audios))
                    else:
                        adjusted_batch_size = min(batch_size, len(processed_audios))
                    
                    pipeline_kwargs["batch_size"] = adjusted_batch_size

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                try:
                    results = self.pipe(processed_audios, **pipeline_kwargs)
                    
                    if return_timestamps:
                        for result in results:
                            self._fix_timestamps(result, return_timestamps)
                        
                        return results
                    else:
                        return [result["text"] for result in results]
                except Exception as e:
                    logger.warning(f"Error in batch processing: {str(e)}. Falling back to individual processing.")
                    results = []
                    for i, audio_item in enumerate(processed_audios):
                        try:
                            clean_kwargs = {k: v for k, v in generate_kwargs.items() if k != "attn_implementation"}
                            
                            result = self.pipe(
                                audio_item, 
                                return_timestamps=False,
                                generate_kwargs=clean_kwargs
                            )
                            
                            text = result["text"]
                            processed_result = self._create_dummy_result(text, return_timestamps)
                            
                            results.append(processed_result if return_timestamps else text)
                            logger.info(f"Successfully processed item {i} with fallback method")
                            
                        except Exception as e2:
                            logger.error(f"Error processing item {i}: {str(e2)}")
                            if return_timestamps == "word":
                                results.append({"error": str(e2), "text": "", "chunks": [], "words": []})
                            elif return_timestamps:
                                results.append({"error": str(e2), "text": "", "chunks": []})
                            else:
                                results.append("")
                    return results
            else:
                processed_audio = self._preprocess_audio(audio, sample_rate)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                try:
                    result = self.pipe(
                        processed_audio, 
                        return_timestamps=return_timestamps,
                        generate_kwargs=generate_kwargs
                    )
                    
                    result = self._fix_timestamps(result, return_timestamps)
                    
                    if return_timestamps:
                        return result
                    else:
                        return result["text"]
                except Exception as e:
                    logger.warning(f"Error with default settings: {str(e)}. Trying with modified settings.")
                    
                    try:
                        clean_kwargs = {k: v for k, v in generate_kwargs.items() 
                                      if k != "return_timestamps" and k != "attn_implementation"}
                        
                        result = self.pipe(
                            processed_audio,
                            return_timestamps=False,
                            generate_kwargs=clean_kwargs
                        )
                        
                        text = result["text"]
                        dummy_result = self._create_dummy_result(text, return_timestamps)
                        
                        if return_timestamps:
                            logger.info(f"Returning transcription with dummy {return_timestamps}-level timestamps due to timestamp generation failure")
                            return dummy_result
                        else:
                            return result["text"]
                    except Exception as e2:
                        logger.error(f"Failed with both approaches: {str(e2)}")
                        if return_timestamps:
                            return {"error": str(e2), "text": ""}
                        else:
                            return ""
                
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            if is_batch:
                return [{"error": str(e), "text": ""}]
            else:
                return {"error": str(e), "text": ""}
