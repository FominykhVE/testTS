import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pyAudioAnalysis')))

from pyAudioAnalysis import audioTrainTest as aT

def predict_gender(audio_path):
    try:
        model_path = r"./pyAudioAnalysis/pyAudioAnalysis/data/models/svm_rbf_speaker_male_female"
        Result, P, classNames = aT.file_classification(audio_path, model_path, "svm_rbf")
        gender = classNames[int(Result)]
        return gender
    except Exception as e:
        print("Ошибка gender-классификации:", e)
        return "unknown"
