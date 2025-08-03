import tempfile
import os
from pydub import AudioSegment
from .asr import recognize_speech
from .gender import predict_gender

def mp3_to_dialog_json(mp3_path, model_path="model"):
    audio = AudioSegment.from_mp3(mp3_path)
    audio = audio.set_frame_rate(16000).set_channels(1)

    wav_path = tempfile.mktemp(suffix=".wav")
    audio.export(wav_path, format="wav")

    raw_results = recognize_speech(wav_path, model_path)
    os.remove(wav_path)

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
    speaker_audio = {s: AudioSegment.empty() for s in speakers}
    total_time = {s: 0 for s in speakers}
    speaker_index = 0

    for s in segments:
        speaker = speakers[speaker_index % 2]
        speaker_audio[speaker] += s["audio_fragment"]
        total_time[speaker] += s["end"] - s["start"]
        speaker_index += 1

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
        is_raised = s["volume"] > -25

        dialog.append({
            "source": speaker,
            "text": s["text"],
            "duration": dur,
            "raised_voice": is_raised,
            "gender": genders.get(speaker, "unknown")
        })
        speaker_index += 1

    return {
        "dialog": dialog,
        "result_duration": total_time
    }
