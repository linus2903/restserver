import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import Audio, load_dataset, Dataset
import io
import librosa
import numpy as np

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"
model_path = "./models/openaiwhisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_path)

forced_decoder_ids = processor.get_decoder_prompt_ids(language="german")

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

def transcribe_path(audio_file_name):
    audio_path = "./uploaded_audio/" + audio_file_name
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

    result = pipe(sample, generate_kwargs={"max_new_tokens": 128, "forced_decoder_ids": forced_decoder_ids})
    print(result)
    return result

def transcribe_file(audio_file):
    # Datei in Bytes einlesen
    audio_bytes = audio_file.file.read()
    audio_stream = io.BytesIO(audio_bytes)
    audio_array, sr = librosa.load(audio_stream, sr=16000)
    audio_array = np.array(audio_array)

    inputs = processor(
    audio_array,
    sampling_rate=sr,
    return_tensors="pt",
    truncation=False,
    max_length=3000,
    padding="max_length",
    return_attention_mask=True,
    )
    inputs = inputs.to(device, dtype=torch_dtype)

    result = pipe(audio_array, generate_kwargs={"max_new_tokens": 128, "forced_decoder_ids": forced_decoder_ids})
    print(result)
    return result