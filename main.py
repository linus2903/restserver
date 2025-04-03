﻿from fastapi import FastAPI
import uvicorn
from router import *

# Erstelle eine FastAPI-App
app = FastAPI(title="Audio Analyser")
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

