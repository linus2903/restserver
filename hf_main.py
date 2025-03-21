import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import Audio, load_dataset, Dataset
import librosa

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"
model_path = "./models/openaiwhisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

audio_path = "./samples/common_voice_de_17299389.mp3"
dataset = Dataset.from_dict({"audio": [audio_path]})

#dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
dataset = dataset.cast_column("audio", Audio(processor.feature_extractor.sampling_rate))
sample = dataset[0]["audio"]

pipe = pipeline(
    "automatic-speech-recognition", 
    model=model, 
    tokenizer=processor.tokenizer, 
    feature_extractor=processor.feature_extractor, 
    chunk_length_s=30, 
    batch_size=16, 
    return_timestamps=True, 
    torch_dtype=torch_dtype, 
    device=device,
    )

inputs = processor(
    sample["array"],
    sampling_rate=sample["sampling_rate"],
    return_tensors="pt",
    truncation=False,
    max_length=3000, 
    padding="max_length",
    return_attention_mask=True,
)
inputs = inputs.to(device, dtype=torch_dtype)

gen_kwargs = {
    "max_new_tokens": 128,
}

result = pipe(sample, **gen_kwargs)

print(result)
