import numpy as np
import pyaudio
import time
import librosa
import librosa.display
import sklearn
import os
import nltk.classify
from nltk.classify import NaiveBayesClassifier
import random
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
import statistics


def get_features(word, file_num, is_training=False):
    file_type = 'test'
    if is_training:
        file_type = 'train'
    y, sr = librosa.load('./{}_files/{}{}.wav'.format(file_type, word, file_num))

    # Set the hop length; at 22050 Hz, 512 samples ~= 23ms
    hop_length = 512

    # Separate harmonics and percussives into two waveforms
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # Beat track on the percussive signal
    tempo, beat_frames = librosa.beat.beat_track(y=y_percussive,
                                                 sr=sr)

    # Compute MFCC features from the raw signal
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)

    # And the first-order differences (delta features)
    mfcc_delta = librosa.feature.delta(mfcc)

    # Stack and synchronize between beat events
    # This time, we'll use the mean value (default) instead of median
    beat_mfcc_delta = librosa.util.sync(np.vstack([mfcc, mfcc_delta]),
                                        beat_frames)

    # Compute chroma features from the harmonic signal
    chromagram = librosa.feature.chroma_cqt(y=y_harmonic,
                                            sr=sr)

    # Aggregate chroma features between beat events
    # We'll use the median value of each feature between beat frames
    beat_chroma = librosa.util.sync(chromagram,
                                    beat_frames,
                                    aggregate=np.median)

    # Finally, stack all beat-synchronous features together
    beat_features = np.vstack([beat_chroma, beat_mfcc_delta])
    features = {}

    for i in range(len(beat_features)):
        features['beat_feature' + str(i)] = beat_features[i][0]

    return features


class OurRecognizer(object):
    def __init__(self, model=NaiveBayesClassifier):
        self.model = model
        self.val_data = []
        self.train_data = []

    def train(self, word):
        num_files = len(os.listdir(os.getcwd() + '/train_files'))
        data = []
        for i in range(num_files):
            if i % 2 == 1:
                data.append((get_features(word, i, True), 0))
            else:
                data.append((get_features(word, i, True), 1))
        split = int(num_files * 0.2)
        self.train_data, self.val_data = data[split:], data[:split]
        random.shuffle(self.train_data)
        # Train the model
        self.model = self.model.train(self.train_data)

    def classify_correct(self, word, file_num):
        beat_features = get_features(word, file_num)
        correct = self.model.classify(featureset=beat_features)
        if correct == 1:
            return 'Correct! Good job!'
        else:
            return 'Uh-oh, the key syllable was not found'

    def accuracy(self):
        return nltk.classify.accuracy(self.model, self.val_data)
