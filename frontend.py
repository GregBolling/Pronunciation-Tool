import speech_recognition as speech_recog
import pyaudio
from nltk.corpus import brown
import random
import flask
from flask import Flask, session
import backend
from subprocess import run, PIPE

app = Flask(__name__)
app.config.from_mapping(
        SECRET_KEY='dev'
    )
sent = ""
mic = speech_recog.Microphone(device_index=0)
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
        <p align="center" >
            <a href=input >
                <button class=grey>
                    Yes
                </button>
            </a>
        </p>
    '''


def input():
    return flask.render_template('index.html')


@app.route("/audio", methods=['POST'])
def audio():
    with open('/tmp/audio.wav', 'wb') as f:
        f.write(flask.request.data)
    proc = run(['ffprobe', '-of', 'default=noprint_wrappers=1', '/tmp/audio.wav'], text=True, stderr=PIPE)
    return proc.stderr


@app.route("/input")
def input_old():
    sent = evaluation_sents[session['index']]
    return '''
            <h3 align="center">Please read the following sentence: {}</h3>
            <p align="center">
                <a href=recognizer >
                    <button class=grey>
                        Press to record
                    </button>
                </a>
            </p>
            '''.format(sent)


@app.route('/recognizer')
def recognizer_old():
    with mic as audio_file:
        recog.adjust_for_ambient_noise(audio_file)
        audio_data = recog.listen(audio_file)

        print("Converting Speech to Text...")

        result = backend.phoneme_recognizer(audio_data)
        session['index'] += 1
        if session['index'] < len(evaluation_sents):
            return '''
                <h3 align="center">You said the following: {}</h3>
                    <p align="center">
                        <a href=input >
                            <button class=grey >
                                Next
                            </button>
                        </a>
                    </p>
                '''.format(result)
        else:
            session['index'] = 0
            return '''
            <h3 align="center">You said the following: {}</h3>
                <p align="center">
                    <a href=input >
                        <button class=grey >
                            Try again?
                        </button>
                    </a>
                </p>
            '''.format(result)


if __name__ == "__main__":
    app.run()
