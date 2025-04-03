from fastapi import FastAPI, File, UploadFile, APIRouter
import shutil
import os
import uvicorn
#import bart
from pydantic import BaseModel
from transkription_service import transcribe_path, transcribe_file

router = APIRouter()

UPLOAD_DIR = "uploaded_audio"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.get("/")
async def root():
    return {"message": "Hallo, Server läuft!"}

@router.get("/summarize/{text}")
async def summarizeText(text: str):
    return "deaktiviert"
    #return bart.summarize_with_bart(text)

@router.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"filename": file.filename}

@router.get("/transcribe/{file}")
async def transcribeFromPath(file: str):
    return transcribe_path(file)

@router.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    return transcribe_file(file)