from vosk import Model, KaldiRecognizer
import wave
import json

def recognize_speech(wav_path, model_path):
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
    return raw_results
