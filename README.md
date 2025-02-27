# 🌍 FastAPI REST-Server für Datei-Uploads 🚀

## 🛠 Voraussetzungen  
### ✅ **Python installieren**  
Stelle sicher, dass **Python 3.10 oder 3.11** installiert ist (Python 3.13 hat Probleme mit `whisper`).  

Überprüfe die Version mit:  
```sh
python --version
```
dependencies installieren:
```sh
run pip install -r requirements.txt
```

## 🏃‍♂️ **Server starten dev**
```sh
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

```

### swagger api im Browser öffnen: 
http://127.0.0.1:8000/docs
