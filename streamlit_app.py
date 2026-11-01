"""SoundSense Streamlit frontend and inference using direct Librosa features."""
from pathlib import Path
import hashlib
import io
import json
import math
import time
import warnings

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import resample_poly
import streamlit as st
import tensorflow as tf
import librosa

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'soundsense_head.keras' 
SAMPLE_RATE = 16000
SAMPLES = 80000
MAX_BYTES = 10 * 1024 * 1024
DATASET_URL = 'https://github.com/karolpiczak/ESC-50'

st.set_page_config(page_title='SoundSense | Environmental Sound Classification',
                   page_icon='🎧', layout='wide', initial_sidebar_state='expanded')

def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))

@st.cache_resource(show_spinner=False)
def load_trained_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError('soundsense_head.keras is missing in your GitHub repository.')
    return tf.keras.models.load_model(str(MODEL_PATH))

def preprocess_wav(content):
    if not content or len(content) > MAX_BYTES:
        raise ValueError('Choose a nonempty WAV file of 10 MB or less.')
    if len(content) < 12 or content[:4] not in (b'RIFF', b'RIFX') or content[8:12] != b'WAVE':
        raise ValueError('This is not a valid WAV file. Export as WAV first.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rate, signal = wavfile.read(io.BytesIO(content))
    except Exception as exc:
        raise ValueError('Could not decode this WAV file.') from exc
    if signal.ndim not in (1, 2) or len(signal) == 0:
        raise ValueError('The WAV recording contains no usable audio samples.')
    if signal.ndim == 2:
        signal = signal.mean(axis=1)
    if rate != SAMPLE_RATE:
        factor = math.gcd(int(rate), SAMPLE_RATE)
        signal = resample_poly(signal, SAMPLE_RATE // factor, int(rate) // factor)
    signal = np.clip(signal[:SAMPLES], -1.0, 1.0)
    rms = float(np.sqrt(np.mean(signal ** 2)))
    signal = np.pad(signal, (0, max(0, SAMPLES - len(signal)))).astype(np.float32)
    return signal, {'original_sample_rate': int(rate), 'original_duration': round(len(signal)/SAMPLE_RATE, 3),
                    'analysed_seconds': 5.0, 'rms': rms, 'sample_rate': SAMPLE_RATE, 'samples': SAMPLES}

def classify(waveform, audio_info, class_names):
    started = time.perf_counter()
    model = load_trained_model()
    
    mel = librosa.feature.melspectrogram(y=waveform, sr=SAMPLE_RATE, n_mels=32, n_fft=2048, hop_length=2500)
    features = librosa.power_to_db(mel, ref=np.max).flatten()
    if len(features) >= 1024:
        feature_vector = features[:1024]
    else:
        feature_vector = np.pad(features, (0, 1024 - len(features)))
    
    input_data = np.expand_dims(feature_vector, axis=0)
    output = model.predict(input_data)
    
    scores = output[0]
    winner = int(scores.argmax())
    ranking = sorted([{'label': label, 'display_label': label.replace('_', ' ').title(),
                       'score': float(scores[i])} for i, label in enumerate(class_names)],
                      key=lambda row: row['score'], reverse=True)
    return {'label': class_names[winner], 'display_label': ranking[0]['display_label'],
            'score': float(scores[winner]), 'scores': ranking,
            'class_names': list(class_names), 'probabilities': [float(x) for x in scores],
            'audio': audio_info, 'inference_ms': round((time.perf_counter() - started) * 1000, 1)}

def main():
    st.title('Hear it. Understand it.')
    st.write('Meet **SoundSense**. Upload an environmental sound.')

    class_names = ['dog', 'rooster', 'pig', 'cow', 'frog', 'cat', 'hen', 'insects', 'sheep', 'crow']
    class_json_path = ROOT / 'class_names.json'
    if class_json_path.is_file():
        try:
            class_names = read_json(class_json_path)
        except Exception:
            pass

    uploaded = st.file_uploader('Choose a WAV recording', type=['wav'])
    content, filename = None, None
    if uploaded is not None:
        content, filename = uploaded.getvalue(), uploaded.name
        try:
            waveform, audio_info = preprocess_wav(content)
            st.audio(content, format='audio/wav')
            if st.button('Classify sound →', type='primary'):
                with st.spinner('Listening...'):
                    res = classify(waveform, audio_info, class_names)
                    st.success(f"Result: **{res['display_label']}** ({res['score']*100:.1f}% score)")
        except Exception as e:
            st.error(str(e))

if __name__ == '__main__':
    main()
