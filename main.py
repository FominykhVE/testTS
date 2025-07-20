import sys
import os
import wave
import json
import tempfile
import requests
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Optional
from pydub import AudioSegment
from vosk import Model, KaldiRecognizer
from collections import Counter

sys.path.insert(0, os.path.abspath("pyAudioAnalysis"))

from pyAudioAnalysis import audioTrainTest as aT 

app = FastAPI()

import os
from pyAudioAnalysis import audioTrainTest as aT


def predict_gender(audio_path):
    try:
        model_path = r".\pyAudioAnalysis\pyAudioAnalysis\data\models\svm_rbf_speaker_male_female"
        Result, P, classNames = aT.file_classification(audio_path, model_path, "svm_rbf")
        gender = classNames[int(Result)]
        return gender
    except Exception as e:
        print("Ошибка gender-классификации:", e)
        return "unknown"


def mp3_to_dialog_json(mp3_path, model_path="model"):
    if not os.path.exists(model_path):
        raise Exception("Модель VOSK не найдена: " + model_path)

    audio = AudioSegment.from_mp3(mp3_path)
    audio = audio.set_frame_rate(16000)
    audio = audio.set_channels(1)

    wav_path = tempfile.mktemp(suffix=".wav")
    audio.export(wav_path, format="wav")

    wf = wave.open(wav_path, "rb")

    model = Model(model_path)
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)

    raw_results = []
    while True:
        chunk = wf.readframes(4000)
        if len(chunk) == 0:
            break
        if rec.AcceptWaveform(chunk):
            raw_results.append(json.loads(rec.Result()))
    raw_results.append(json.loads(rec.FinalResult()))

    
    segments = []
    for r in raw_results:
        if "result" not in r:
            continue
        wordlist = r["result"]
        if not wordlist:
            continue
        s_time = wordlist[0]["start"]
        e_time = wordlist[-1]["end"]
        line = " ".join([w["word"] for w in wordlist])
        frag = audio[int(s_time * 1000):int(e_time * 1000)]
        volume = frag.dBFS
        segments.append({
            "start": s_time,
            "end": e_time,
            "text": line,
            "volume": volume,
            "audio_fragment": frag
        })

    speakers = ["receiver", "transmitter"]
    speaker_audio = {
        "receiver": AudioSegment.empty(),
        "transmitter": AudioSegment.empty()
    }
    speaker_index = 0
    total_time = {"receiver": 0, "transmitter": 0}

    
    for s in segments:
        speaker = speakers[speaker_index % 2]
        speaker_audio[speaker] += s["audio_fragment"]
        total_time[speaker] += s["end"] - s["start"]
        speaker_index += 1

    # Пол
    genders = {}
    for spk, audio_seg in speaker_audio.items():
        temp_path = tempfile.mktemp(suffix=".wav")
        audio_seg.export(temp_path, format="wav")
        genders[spk] = predict_gender(temp_path)
        os.remove(temp_path)
    
    dialog = []
    speaker_index = 0
    for s in segments:
        dur = round(s["end"] - s["start"], 2)
        speaker = speakers[speaker_index % 2]
        is_raised = False
        if s["volume"] > -25:
            is_raised = True

        dialog.append({
            "source": speaker,
            "text": s["text"],
            "duration": dur,
            "raised_voice": is_raised,
            "gender": genders.get(speaker, "unknown")
        })
        speaker_index = speaker_index + 1

    return {
        "dialog": dialog,
        "result_duration": total_time
    }


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
    test_model = "./models/vosk-model-small-ru-0.22"
    test_file = "testDialog.mp3" # Файл MP3 в папке проекта

    res = mp3_to_dialog_json(test_file, test_model)
    print(json.dumps(res, indent=4, ensure_ascii=False))
