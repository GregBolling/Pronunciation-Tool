import speech_recognition as speech_recog
import pyaudio
from nltk.corpus import brown
import random
import os
from pocketsphinx import Decoder, get_model_path


def generate_random_sent():
    corpus = brown.sents()
    rand = random.randint(0, len(corpus))
    return corpus[rand]


if __name__ == '__main__':
    mic = speech_recog.Microphone()
    recog = speech_recog.Recognizer()
    model_path = get_model_path()

    choice = input("Ready to start?(y/n): ")
    while choice == 'y' or choice == 'Y':
        with mic as audio_file:
            print("Say: Time can be a good teacher")

            recog.adjust_for_ambient_noise(audio_file)
            audio_data = recog.listen(audio_file)

            print("Converting Speech to Text...")

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

            print('Best hypothesis segments:', [seg.word for seg in decoder.seg()])

            print("You said: " + recog.recognize_sphinx(audio_data))
            choice = input("Wanna play again?(y/n)")
