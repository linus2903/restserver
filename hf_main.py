import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
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

audio_path = "./samples/aufzeichnung.wav"
dataset = Dataset.from_dict({"audio": [audio_path]})

#dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
dataset = dataset.cast_column("audio", Audio(processor.feature_extractor.sampling_rate))
sample = dataset[0]["audio"]

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
    "max_new_tokens": 448,
    "num_beams": 1,
    "condition_on_prev_tokens": False,
    "compression_ratio_threshold": 1.35,  # zlib compression ratio threshold (in token space)
    "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    "logprob_threshold": -1.0,
    "no_speech_threshold": 0.6,
    "return_timestamps": True,
}

pred_ids = model.generate(**inputs, **gen_kwargs)
pred_text = processor.batch_decode(pred_ids, skip_special_tokens=True, decode_with_timestamps=False)

print(pred_text)
