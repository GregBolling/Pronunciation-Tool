from nltk.corpus import brown
import random
import flask
from flask import Flask
import backend
from subprocess import run, PIPE

app = Flask(__name__)
app.config.from_mapping(
        SECRET_KEY='dev'
    )
sent = ""
evaluation_sents = ['time', 'dog', 'can', 'teacher', 'fat', 'red', 'bat', 'cheap', 'shark', 'shop', 'bake']
recog = backend.CMURecognizer(evaluation_sents)


def generate_random_sent():
    corpus = brown.sents()
    rand = random.randint(0, len(corpus))
    return corpus[rand]


@app.route("/") # take note of this decorator syntax, it's a common pattern
def homepage():
    return '''
        <h3 align="center">Welcome to our Pronunciation Tool! Which evaluation model would you like to use?</h3>
        <p align="center" >
            <a href=input_cmu >
                <button class=grey>
                    CMU PocketSphinx
                </button>
            </a>
        </p>
        <p align="center" >
            <a href=input >
                <button class=grey>
                    Our Specialized Model
                </button>
            </a>
        </p>
    '''


@app.route("/input")
def input_ours():
    return flask.render_template('index.html')


@app.route("/audio", methods=['POST'])
def audio():
    return backend.classify_correct(flask.request.data)


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
