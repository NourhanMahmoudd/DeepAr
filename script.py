from datasets import load_dataset , Audio
from whisper_arabic.model import ArabicWhisper  # If ArabicWhisper is in model.py
import torch
import os
from tqdm import tqdm


# Load the dataset
dataset = load_dataset("AtharvA7k/ClArTTS", split="test")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
# Initialize the model
model = ArabicWhisper()

# Create output directory for results
output_dir = "transcription_results"
os.makedirs(output_dir, exist_ok=True)

# Process each audio file in the dataset
results = []

for idx, item in enumerate(tqdm(dataset, desc="Transcribing")):

        # Get the audio array
        audio_array = torch.tensor(item['audio']['array'])

        
        # Transcribe the audio
        transcription = model.transcribe(audio_array)

        
        # Store results
        result = {
            'id': idx,
            'transcription': transcription,
            'original_text': item.get('text', 'No original text available')
        }
        results.append(result)
        
        # Save individual result
        with open(os.path.join(output_dir, f"transcription_{idx}.txt"), 'w', encoding='utf-8') as f:
            f.write(f"Original: {result['original_text']}\n")
            f.write(f"Transcribed: {result['transcription']}\n")
        
        if idx > 10:
            break


# Save all results to a summary file
with open(os.path.join(output_dir, "summary.txt"), 'w', encoding='utf-8') as f:
    f.write("Summary of Transcriptions:\n\n")
    for result in results:
        f.write(f"Sample {result['id']}:\n")
        f.write(f"Original: {result['original_text']}\n")
        f.write(f"Transcribed: {result['transcription']}\n")
        f.write("-" * 50 + "\n")

print(f"Transcription completed. Results saved in {output_dir}")

# Print some statistics
total_samples = len(results)
print(f"\nTotal samples processed: {total_samples}")