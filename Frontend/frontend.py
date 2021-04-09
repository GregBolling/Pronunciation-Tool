from nltk.corpus import brown
import random
import flask
from flask import Flask
import cmu_backend
import our_backend
import os

app = Flask(__name__)
app.config.from_mapping(
        SECRET_KEY='dev'
    )
sent = ""
evaluation_sents = ['time', 'dog', 'can', 'teacher', 'fat', 'red', 'bat', 'cheap', 'shark', 'shop', 'bake']
recog = cmu_backend.CMURecognizer(evaluation_sents)


def generate_random_sent():
    corpus = brown.sents()
    rand = random.randint(0, len(corpus))
    return corpus[rand]


@app.route("/") # take note of this decorator syntax, it's a common pattern
def homepage():
    return flask.render_template('hometitle.html')


@app.route("/input")
def input_ours():
    return flask.render_template('index.html')


@app.route("/audio", methods=['POST'])
def audio():
    num_files = len(os.listdir(os.getcwd() + '/Backend/test_files'))
    with open('./Backend/test_files/audio{}.wav'.format(num_files), 'wb') as f:
        f.write(flask.request.data)

    return our_backend.OurRecognizer().classify_correct(num_files)


@app.route("/input_cmu")
def input_cmu():
    sent = recog.get_current_sent()
    return '''
            <h3 align="center">Please read the following sentence: {}</h3>
            <p align="center">
                <a href=record >
                    <button class=grey>
                        Press to record
                    </button>
                </a>
            </p>
            '''.format(sent)


@app.route('/record')
def record():
    return recog.record()


@app.route('/recognizer')
def recognizer_old():
    return recog.phoneme_recognizer()


if __name__ == "__main__":
    app.run()
