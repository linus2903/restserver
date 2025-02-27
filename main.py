from fastapi import FastAPI, File, UploadFile
import shutil
import os
import whisper
import uvicorn

# Erstelle eine FastAPI-App
app = FastAPI()

UPLOAD_DIR = "uploaded_audio"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Beispiel-Endpunkt: Root-Route
@app.get("/")
async def root():
    return {"message": "Hallo, FastAPI läuft!"}

# Beispiel-Endpunkt: Echo-Route
@app.get("/echo/{text}")
async def echo(text: str):
    return {"echo": text}

# Beispiel-Endpunkt: POST mit JSON-Daten
from pydantic import BaseModel

@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    model = whisper.load_model("turbo")
    result = model.transcribe(file_path)
    print(result["text"])

    return {"filename": file.filename, "message": result["text"]}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

