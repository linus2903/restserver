from datasets import Audio, load_dataset, Dataset
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, WhisperProcessor
import time
from pathlib import Path
import pandas as pd
import os


model_path = "./models/openaiwhisper-large-v3-turbo"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

def load_pipeline():
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_path)

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
    return pipe

def load_dataset2(folder_path):
    #dataset = load_dataset("audiofolder", data_dir=folder_path)
    dataset = load_dataset(folder_path)

    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    return dataset

def transcribe_dataset(dataset):
    dataset_train = dataset["train"]
    pipe = load_pipeline()

    transcriptions = []
    for row in dataset_train:
        audio_file = row["audio"]
        start_time = time.time()
        result = pipe(audio_file["array"], generate_kwargs={"language": "german", "task": "transcribe"})
        duration = time.time() - start_time

        _dict = {
            "transcribed text": result["text"],
            "file": os.path.basename(audio_file["path"]),
            "original length (sec)": result["chunks"][-1]["timestamp"][-1],
            "computation time (sec)": duration,
            "transcription lösung": row["transcription"]
        }

        transcriptions.append(_dict)
    return transcriptions

def create_table(transcriptions):
    df = pd.DataFrame(transcriptions)
    print("\n📜 **Transkriptions-Ergebnisse:**")
    print(df.to_string(index=False))


create_table(transcribe_dataset(load_dataset2("C:/SPU\pwe/Neuer Ordner/restserver/samples")))