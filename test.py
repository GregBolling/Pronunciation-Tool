import speech_recognition as speech_recog
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import wave
from gtts import gTTS
import our_backend
from pydub import AudioSegment


mic = speech_recog.Microphone(device_index=0)
recog = speech_recog.Recognizer()
evaluation_sents = ['time', 'bad', 'can', 'teacher', 'fat', 'red', 'bat', 'cheap', 'shark', 'shop', 'bake']

if __name__ == '__main__':
    word = evaluation_sents[0]
    data = []
    for i in range(1000):
        if i % 4 == 3:
            tts = gTTS('kime')
        else:
            tts = gTTS('time')
        tts.save('test_files/' + word + '.mp3')
        src = 'test_files/' + word + '.mp3'
        dst = 'test_files/' + word + str(i) + '.wav'
        sound = AudioSegment.from_mp3(src)
        sound.export(dst, format="wav")
