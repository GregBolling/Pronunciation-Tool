from pocketsphinx import Decoder, get_model_path
import os


def phoneme_recognizer(audio_data):
    model_path = get_model_path()
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

    return [seg.word for seg in decoder.seg()]


def classify_correct(audio_data):
    return 0
