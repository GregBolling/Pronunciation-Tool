import numpy as np
import pyaudio
import time
import librosa
import librosa.display
from nltk.classify import NaiveBayesClassifier
import sklearn
import os
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential


class OurRecognizer(object):
    def __init__(self):
        # Call neural network API
        self.model = Sequential()
        # Apply linear activation function to input layer
        # Generate hidden layer with 38 nodes, the same as the input layer
        self.model.add(Dense(units=38, activation='linear', input_dim=38))
        self.model.add(Dense(units=14, activation='linear'))
        # Apply linear activation function to hidden layer
        # Generate output layer with 14 nodes
        self.model.add(Dense(units=2, activation='linear'))
        # Compile the model
        self.model.compile(optimizer='adam',
                      loss='mean_squared_error',
                      metrics=['accuracy'])
        # self.train()

    def train(self):
        num_files = len(os.listdir(os.getcwd() + '/test_files'))
        data = []
        for i in range(num_files):
            data.append(self.get_features(i))

        split = int(num_files * 0.2)
        train_data, val_data = data[split:], data[:split]
        # Train the model
        num_epochs = 10
        batch_size = 256
        self.model = self.model.fit(x=train_data, y=train_data,
                            epochs=num_epochs,
                            batch_size=batch_size,
                            shuffle=True,
                            validation_data=(val_data, val_data),
                            verbose=1)

    def get_features(self, file_num):
        y, sr = librosa.load('./test_files/audio{}.wav'.format(file_num))

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
        scaled_beat_features = sklearn.preprocessing.scale(beat_features, axis=0)
        features = {}

        for i in range(len(scaled_beat_features)):
            features['beat_feature' + str(i)] = scaled_beat_features[i]

        return scaled_beat_features

    def classify_correct(self, file_num):
        beat_features = self.get_features(file_num)
        # self.model.predict(beat_features)

        return str(beat_features)
