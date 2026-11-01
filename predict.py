from pathlib import Path
import argparse, json, math
import numpy as np
import tensorflow as tf
from scipy.io import wavfile
from scipy.signal import resample_poly
SAMPLE_RATE = 16000
SAMPLES = 80000

def load_audio(path):
    rate, signal = wavfile.read(path)
    if signal.dtype == np.int16:
        signal = signal.astype(np.float32) / 32768.0
    elif signal.dtype == np.int32:
        signal = signal.astype(np.float32) / 2147483648.0
    elif signal.dtype == np.uint8:
        signal = (signal.astype(np.float32) - 128.0) / 128.0
    elif np.issubdtype(signal.dtype, np.floating):
        signal = signal.astype(np.float32)
    else:
        raise ValueError('Unsupported WAV encoding.')
    if signal.ndim == 2:
        signal = signal.mean(axis=1)
    if signal.ndim != 1 or len(signal) == 0 or not np.isfinite(signal).all():
        raise ValueError('Invalid or empty audio.')
    if rate != SAMPLE_RATE:
        factor = math.gcd(int(rate), SAMPLE_RATE)
        signal = resample_poly(signal, SAMPLE_RATE // factor, int(rate) // factor)
    signal = np.clip(signal[:SAMPLES], -1.0, 1.0)
    return np.pad(signal, (0, max(0, SAMPLES - len(signal)))).astype(np.float32)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify one WAV file with the complete SoundSense model.')
    parser.add_argument('audio', help='Path to a WAV recording')
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent
    classes = json.loads((folder / 'class_names.json').read_text())
    model = tf.saved_model.load(str(folder / 'soundsense_saved_model'))
    scores = model.signatures['serving_default'](waveform=tf.constant(load_audio(args.audio)))['probabilities'].numpy()[0]
    print(json.dumps({'label': classes[int(scores.argmax())], 'model_score': float(scores.max()), 'scores': dict(zip(classes, map(float, scores)))}, indent=2))
