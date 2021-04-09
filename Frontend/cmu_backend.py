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


