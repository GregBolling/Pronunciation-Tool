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
evaluation_sents = [('time', 'T'), ('can', 'K'), ('teacher', 'ER'), ('fat', 'F'), ('red', 'R'), ('bat', 'B'),
                    ('cheap', 'CH'), ('shark', 'SH'), ('shop', 'SH'), ('bake', 'B')]
recog = cmu_backend.CMURecognizer(evaluation_sents)
our_recog = our_backend.OurRecognizer()
our_recog.train(evaluation_sents[0][0])


@app.route("/") # take note of this decorator syntax, it's a common pattern
def homepage():
    return flask.render_template('hometitle.html')


@app.route("/input")
def input_ours():
    return flask.render_template('index.html')


@app.route("/audio", methods=['POST'])
def audio():
    num_files = len(os.listdir(os.getcwd() + '/test_files'))
    word = evaluation_sents[0][0]
    with open('./test_files/{}{}.wav'.format(word, num_files), 'wb') as f:
        f.write(flask.request.data)

    return our_recog.classify_correct(word, num_files)


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
