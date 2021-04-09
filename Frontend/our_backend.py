import numpy as np
import pyaudio
import time
import librosa
import librosa.display
from nltk.classify import NaiveBayesClassifier


class OurRecognizer(object):
    def __init__(self):
        self.model = NaiveBayesClassifier
        self.beat_features = None

    def classify_correct(self, num_files):
        y, sr = librosa.load('./Backend/test_files/audio{}.wav'.format(num_files))

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
        self.beat_features = np.vstack([beat_chroma, beat_mfcc_delta])

        librosa.display.specshow(mfcc, sr=sr, x_axis='time')
        return str(self.beat_features)