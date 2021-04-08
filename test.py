import speech_recognition as speech_recog
import backend
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import librosa.display
import wave


mic = speech_recog.Microphone(device_index=0)
recog = speech_recog.Recognizer()
evaluation_sents = ['time', 'bad', 'can', 'teacher', 'fat', 'red', 'bat', 'cheap', 'shark', 'shop', 'bake']

if __name__ == '__main__':
    with mic as audio_file:
        recog.adjust_for_ambient_noise(audio_file)
        audio_data = recog.listen(audio_file)

        w = wave.open("test.wav", 'wb')
        frame = audio_data.frame_data
        w.setparams((1, audio_data.sample_width))
        w.writeframes(audio_data.get_wav_data())
        w.close()

        sr = audio_data.sample_rate

        plt.figure(figsize=(20, 5))
        array = np.frombuffer(audio_data.get_raw_data(), dtype=np.float32)
        librosa.display.waveplot(array, sr=sr)
