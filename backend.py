from pocketsphinx import Decoder, get_model_path
import os
import numpy as np
import pyaudio
import time
import librosa


def phoneme_recognizer(audio_data):
    model_path = get_model_path()
    raw_data = audio_data.get_raw_data(convert_rate=16000,
                                       convert_width=2)  # the included language models require audio to be 16-bit mono 16 kHz in little-endian format

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
                        True)  # process audio data with recognition enabled (no_search = False), as a full utterance (full_utt = True)
    decoder.end_utt()  # stop utterance processing

    return [seg.word for seg in decoder.seg()]


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
    return None
