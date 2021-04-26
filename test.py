import speech_recognition as speech_recog
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import wave
from gtts import gTTS
import our_backend
import cmu_backend
import os
import nltk
from pydub import AudioSegment

evaluation_sents = [('time', 'T'), ('can', 'K'), ('teacher', 'ER'), ('fat', 'F'), ('red', 'R'), ('bat', 'B'),
                    ('cheap', 'CH'), ('shark', 'SH'), ('shop', 'SH'), ('bake', 'B')]
r = speech_recog.Recognizer()


def cmu_accuracy(word):
    cmu_recog = cmu_backend.CMURecognizer(evaluation_sents)
    num_files = len(os.listdir(os.getcwd() + '/train_files'))
    correct = 0
    for i in range(num_files):
        is_correct = 1
        if i % 2 == 1:
            is_correct = 0
        train_file = speech_recog.AudioFile('./train_files/{}{}.wav'.format(word, i))
        with train_file as source:
            audio = r.record(source)
            cmu_recog.audio_data = audio
            if cmu_recog.contains_phoneme() == is_correct:
                correct += 1
    accuracy = correct / num_files
    return accuracy


def our_accuracy(word, model):
    our_recog = our_backend.OurRecognizer(model)
    our_recog.train(word)
    return our_recog.accuracy()


if __name__ == '__main__':
    word = evaluation_sents[0][0]
    print("CMU Accuracy: ", cmu_accuracy(word))
    print("Naive Bayes Accuracy: ", our_accuracy(word, nltk.classify.NaiveBayesClassifier))
    print("Decision Tree Accuracy", our_accuracy(word, nltk.classify.DecisionTreeClassifier))
    print("Max Entropy Accuracy", our_accuracy(word, nltk.classify.MaxentClassifier))
