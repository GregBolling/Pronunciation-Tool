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
#our_recog.train(evaluation_sents[0][0])
is_training = False
num_submissions = 0


@app.route("/") # take note of this decorator syntax, it's a common pattern
def homepage():
    return flask.render_template('hometitle.html')


@app.route("/input")
def input_ours():
    global is_training
    is_training = False
    sent = '"' + recog.get_current_sent() + '"'
    return flask.render_template('index.html', sent=sent)


@app.route("/add_data")
def add_data():
    global is_training
    is_training = True
    sent = '"' + recog.get_current_sent() + '" (3 times) and "kime" (once)'
    return flask.render_template('index.html', sent=sent)


@app.route("/audio", methods=['POST'])
def audio():
    if not is_training:
        num_files = len(os.listdir(os.getcwd() + '/test_files'))
        word = evaluation_sents[0][0]
        with open('./test_files/{}{}.wav'.format(word, num_files), 'wb') as f:
            f.write(flask.request.data)

        return our_recog.classify_correct(word, num_files)
    num_files = len(os.listdir(os.getcwd() + '/train_files'))
    word = evaluation_sents[0][0]
    with open('./train_files/{}{}.wav'.format(word, num_files), 'wb') as f:
        f.write(flask.request.data)
    global num_submissions
    num_submissions += 1
    return 'Thank you for submission #' + str(num_submissions)


@app.route("/input_cmu")
def input_cmu():
    sent = recog.get_current_sent()
    return flask.render_template('thecmu.html', sent=sent)


@app.route('/record')
def record():
    return recog.record()


@app.route('/recognizer')
def recognizer_old():
    return recog.phoneme_recognizer()


if __name__ == "__main__":
    app.run()
