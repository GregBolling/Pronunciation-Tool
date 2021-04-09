from pocketsphinx import Decoder, get_model_path
import os
import numpy as np
import pyaudio
import time
import librosa
import speech_recognition as speech_recog


class CMURecognizer(object):
    def __init__(self, sents):
        self.sents = sents
        self.index = 0
        self.audio_data = None

    def get_current_sent(self):
        return self.sents[self.index]

    def record(self):
        mic = speech_recog.Microphone(device_index=0)
        recog = speech_recog.Recognizer()
        with mic as audio_file:
            recog.adjust_for_ambient_noise(audio_file)
            self.audio_data = recog.listen(audio_file)
        return '''
            <h3 align="center">Audio Recorded</h3>
                    <p align="center">
                        <a href=recognizer >
                            <button class=grey>
                                Press to analyze recording
                            </button>
                        </a>
                    </p>
            '''

    def phoneme_recognizer(self):
        self.index += 1
        audio_data = self.audio_data
        model_path = get_model_path()
        raw_data = audio_data.get_raw_data(convert_rate=16000,
                                           convert_width=2)  # the included language models require audio to be
        # 16-bit mono 16 kHz in little-endian format

        # Create a decoder with a certain model
        config = Decoder.default_config()
        config.set_string('-hmm', os.path.join(model_path, 'en-us'))
        config.set_string('-allphone', os.path.join(model_path, 'en-us-phone.lm.dmp'))
        config.set_float('-lw', 2.0)
        config.set_float('-beam', 1e-10)
        config.set_float('-pbeam', 1e-10)
        decoder = Decoder(config)

        decoder.start_utt()  # begin utterance processing
        decoder.process_raw(raw_data, False,
                            True)  # process audio data with recognition enabled (no_search = False), as a full
        # utterance (full_utt = True)
        decoder.end_utt()  # stop utterance processing
        if self.index < len(self.sents):
            return '''
                                <h3 align="center">You said the following: {}</h3>
                                    <p align="center">
                                        <a href=input_cmu >
                                            <button class=grey >
                                                Next
                                            </button>
                                        </a>
                                    </p>
                                '''.format([seg.word for seg in decoder.seg()])
        else:
            self.index = 0
            return '''
                    <h3 align="center">You said the following: {}</h3>
                        <p align="center">
                            <a href=input_cmu >
                                <button class=grey >
                                    Try again?
                                </button>
                            </a>
                        </p>
                    '''.format([seg.word for seg in decoder.seg()])


class AudioHandler(object):
    def __init__(self):
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 44100
        self.CHUNK = 1024 * 2
        self.p = None
        self.stream = None
        self.features = []

    def start(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.FORMAT,
                                  channels=self.CHANNELS,
                                  rate=self.RATE,
                                  input=True,
                                  output=False,
                                  stream_callback=self.callback,
                                  frames_per_buffer=self.CHUNK)
        self.stream.start_stream()

    def stop(self):
        self.stream.close()
        self.p.terminate()

    def callback(self, in_data, frame_count, time_info, flag):
        numpy_array = np.frombuffer(in_data, dtype=np.float32)
        feature = librosa.feature.mfcc(numpy_array)
        return feature, pyaudio.paContinue

    def mainloop(self):
        while self.stream.is_active():  # if using button you can set self.stream to 0 (self.stream = 0), otherwise
            # you can use a stop condition
            time.sleep(2.0)


def classify_correct(audio_data):
    numpy_array = np.frombuffer(audio_data, dtype=np.float32)
    array_no_nan = numpy_array[~np.isnan(numpy_array)]
    feature = librosa.feature.mfcc(array_no_nan, sr=44100)
    return str(feature)
