from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Optional
import tempfile
import requests
import os
from recognizer.segmentation import mp3_to_dialog_json

app = FastAPI()

@app.post("/asr")
async def recognize_asr(file: Optional[UploadFile] = File(None), url: Optional[str] = Form(None)):
    if not file and not url:
        return JSONResponse(content={"error": "не передан файл или URL"}, status_code=400)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_mp3:
        if file:
            temp_mp3.write(await file.read())
        elif url:
            try:
                r = requests.get(url)
                r.raise_for_status()
                temp_mp3.write(r.content)
            except Exception as e:
                return JSONResponse(content={"error": "Ошибка при загрузке: " + str(e)}, status_code=400)

        mp3_file_path = temp_mp3.name

    try:
        model_path = "./models/vosk-model-small-ru-0.22"
        result = mp3_to_dialog_json(mp3_file_path, model_path)
    finally:
        try:
            os.remove(mp3_file_path)
        except:
            pass

    return result

if __name__ == "__main__":
    res = mp3_to_dialog_json("testDialog.mp3", "./models/vosk-model-small-ru-0.22")
    import json
    print(json.dumps(res, indent=4, ensure_ascii=False))
