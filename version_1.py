import speech_recognition as speech_recog
import pyaudio
from nltk.corpus import brown
import random
import os
from pocketsphinx import Decoder, get_model_path
from flask import Flask, session
app = Flask(__name__)
app.config.from_mapping(
        SECRET_KEY='dev'
    )
sent = ""
mic = speech_recog.Microphone()
recog = speech_recog.Recognizer()
evaluation_sents = ['time', 'bad', 'can', 'teacher', 'fat', 'red', 'bat', 'cheap', 'shark', 'shop', 'bake']


def generate_random_sent():
    corpus = brown.sents()
    rand = random.randint(0, len(corpus))
    return corpus[rand]


@app.route("/") # take note of this decorator syntax, it's a common pattern
def homepage():
    session['index'] = 0
    return '''
        <h3 align="center">Welcome! Would you like to get started?</h3>
        <p align="center">
            <a href=input >
                <button class=grey style="height:75px;width:150px">
                    Yes
                </button>
            </a>
        </p>
    '''


@app.route("/input")
def input():
    sent = evaluation_sents[session['index']]
    return '''
            <h3 align="center">Please read the following sentence: {}</h3>
            <p align="center">
                <a href=recognizer >
                    <button class=grey style="height:75px;width:150px">
                        Press to record
                    </button>
                </a>
            </p>
            '''.format(sent)


@app.route('/recognizer')
def recognizer():
    with mic as audio_file:
        model_path = get_model_path()
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

        result = [seg.word for seg in decoder.seg()]
        if session['index'] < len(evaluation_sents):
            session['index'] += 1
            return '''
                <h3 align="center">You said the following: {}</h3>
                    <p align="center">
                        <a href=input >
                            <button class=grey style="height:75px;width:150px">
                                Next
                            </button>
                        </a>
                    </p>
                '''.format(result)
        else:
            index = 0
            return '''
            <h3 align="center">You said the following: {}</h3>
                <p align="center">
                    <a href=input >
                        <button class=grey style="height:75px;width:150px">
                            Try again?
                        </button>
                    </a>
                </p>
            '''.format(result)


if __name__ == "__main__":
    app.run()
