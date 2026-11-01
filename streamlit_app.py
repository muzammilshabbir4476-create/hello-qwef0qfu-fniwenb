"""SoundSense Streamlit frontend and inference.

Run from this folder with: streamlit run streamlit_app.py
Deploy this .py file as the Streamlit Community Cloud entrypoint.
Keep model/, assets/, results/ and notebooks/ beside it in the repository.
"""
from pathlib import Path
import hashlib
import io
import json
import math
import time
import warnings
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import resample_poly
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'model/soundsense.onnx'
SAMPLE_RATE = 16000
SAMPLES = 80000
MAX_BYTES = 10 * 1024 * 1024
DATASET_URL = 'https://github.com/karolpiczak/ESC-50'

st.set_page_config(page_title='SoundSense | Environmental Sound Classification',
                   page_icon='🎧', layout='wide', initial_sidebar_state='expanded')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


@st.cache_resource(show_spinner=False)
def load_model(expected_sha256, class_names):
    """One shared, thread-safe CPU inference session across Streamlit reruns."""
    import onnxruntime as ort
    if not MODEL_PATH.is_file():
        raise FileNotFoundError('model/soundsense.onnx is missing. Upload the complete model folder to GitHub.')
    if hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError('The model checksum does not match model/manifest.json. Restore the supplied model folder, or export the updated model and manifest together.')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    session = ort.InferenceSession(str(MODEL_PATH), sess_options=options,
                                   providers=['CPUExecutionProvider'])
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if len(inputs) != 1 or inputs[0].shape != [SAMPLES]:
        raise ValueError('This app requires the complete SoundSense model with one [80000] waveform input.')
    if len(outputs) != 1 or outputs[0].shape != [1, len(class_names)]:
        raise ValueError('The model output and class_names.json do not agree.')
    return session


def preprocess_wav(content):
    """Match the supplied notebook: float mono -> resample -> crop/clip/pad."""
    if not content or len(content) > MAX_BYTES:
        raise ValueError('Choose a nonempty WAV file of 10 MB or less.')
    if len(content) < 12 or content[:4] not in (b'RIFF', b'RIFX') or content[8:12] != b'WAVE':
        raise ValueError('This is not a valid WAV file. Export MP3 or other formats as WAV first.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rate, signal = wavfile.read(io.BytesIO(content))
    except Exception as exc:
        raise ValueError('Could not decode this WAV file. Please export it again as PCM16 or float WAV.') from exc
    if signal.ndim not in (1, 2) or len(signal) == 0:
        raise ValueError('The WAV recording contains no usable audio samples.')
    if signal.ndim == 2 and not 1 <= signal.shape[1] <= 8:
        raise ValueError('Use mono or stereo audio, with no more than eight channels.')
    if not 8000 <= rate <= 192000:
        raise ValueError('Use a sample rate between 8 kHz and 192 kHz.')
    seconds = len(signal) / rate
    if seconds > 30:
        raise ValueError('Choose a recording no longer than 30 seconds. Only the first five seconds are analysed.')
    if np.issubdtype(signal.dtype, np.signedinteger) and signal.dtype.itemsize == 2:
        signal = signal.astype(np.float32) / 32768.0
    elif np.issubdtype(signal.dtype, np.signedinteger) and signal.dtype.itemsize == 4:
        signal = signal.astype(np.float32) / 2147483648.0
    elif signal.dtype == np.uint8:
        signal = (signal.astype(np.float32) - 128.0) / 128.0
    elif np.issubdtype(signal.dtype, np.floating):
        signal = signal.astype(np.float32)
    else:
        raise ValueError('Unsupported WAV encoding. Use PCM16, PCM24, PCM32 or float WAV.')
    if signal.ndim == 2:
        signal = signal.mean(axis=1)
    if not np.isfinite(signal).all():
        raise ValueError('The recording contains invalid audio samples.')
    if rate != SAMPLE_RATE:
        factor = math.gcd(int(rate), SAMPLE_RATE)
        signal = resample_poly(signal, SAMPLE_RATE // factor, int(rate) // factor)
    signal = np.clip(signal[:SAMPLES], -1.0, 1.0)
    rms = float(np.sqrt(np.mean(signal ** 2)))
    signal = np.pad(signal, (0, max(0, SAMPLES - len(signal)))).astype(np.float32)
    return signal, {'original_sample_rate': int(rate), 'original_duration': round(seconds, 3),
                    'analysed_seconds': min(round(seconds, 3), 5.0), 'rms': rms,
                    'sample_rate': SAMPLE_RATE, 'samples': SAMPLES}


def classify(waveform, audio_info, class_names, expected_sha256):
    started = time.perf_counter()
    session = load_model(expected_sha256, tuple(class_names))
    output = session.run(None, {session.get_inputs()[0].name: waveform})[0]
    if output.shape != (1, len(class_names)) or not np.isfinite(output).all():
        raise ValueError('The model returned an invalid result.')
    scores = output[0]
    if np.any(scores < 0) or not np.isclose(scores.sum(), 1, atol=1e-4):
        raise ValueError('The model output is not a valid softmax score vector.')
    winner = int(scores.argmax())
    ranking = sorted([{'label': label, 'display_label': label.replace('_', ' ').title(),
                       'score': float(scores[i])} for i, label in enumerate(class_names)],
                     key=lambda row: row['score'], reverse=True)
    return {'label': class_names[winner], 'display_label': ranking[0]['display_label'],
            'score': float(scores[winner]), 'scores': ranking,
            'class_names': list(class_names), 'probabilities': [float(x) for x in scores],
            'audio': audio_info, 'inference_ms': round((time.perf_counter() - started) * 1000, 1),
            'score_note': 'Softmax scores are not calibrated probabilities of real-world correctness.'}


def download_file(label, relative_path, download_name=None, mime=None, key=None):
    path = ROOT / relative_path
    if path.is_file():
        st.download_button(label, path.read_bytes(), file_name=download_name or path.name,
                           mime=mime, width='stretch', key=key)
    else:
        st.caption(f'{label}: unavailable')


def main():
    st.markdown('''<style>
      .block-container {max-width:1250px;padding-top:2.5rem;padding-bottom:2.5rem;}
      h1 {letter-spacing:-1.8px;}
      [data-testid="stMetric"] {background:#edf2e6;border:1px solid #dce5d3;
        border-radius:12px;padding:16px;}
      [data-testid="stSidebar"] {border-right:1px solid #dfe6d7;}
      .sound-kicker {font-size:11px;letter-spacing:2px;color:#708565;font-weight:600;}
    </style>''', unsafe_allow_html=True)
    st.markdown('<p class="sound-kicker">ENVIRONMENTAL SOUND CLASSIFICATION</p>', unsafe_allow_html=True)
    st.title('Hear it. Understand it.')
    st.write('Meet **SoundSense**. Upload an everyday sound and let your trained deep learning model do the listening.')

    required = [MODEL_PATH, ROOT / 'model/class_names.json', ROOT / 'model/manifest.json']
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        st.error('Required deployment files are missing: ' + ', '.join(missing))
        st.info('Upload the model folder beside streamlit_app.py in GitHub, keeping the folder structure.')
        st.stop()
    try:
        class_names = read_json(ROOT / 'model/class_names.json')
        manifest = read_json(ROOT / 'model/manifest.json')
        if class_names != manifest['class_names'] or len(class_names) != 10:
            raise ValueError('The class order does not match the SoundSense model.')
        model_sha256 = manifest['sha256']
    except (ValueError, KeyError, TypeError) as exc:
        st.error('Could not read the model specification: ' + str(exc))
        st.stop()

    metrics_path = ROOT / 'results/metrics.json'
    metrics = read_json(metrics_path) if metrics_path.is_file() else {}
    links_path = ROOT / 'project_links.json'
    links = read_json(links_path) if links_path.is_file() else {}

    with st.sidebar:
        st.header('🎧 SoundSense')
        st.caption('Deep Learning Semester Project')
        st.markdown('**Junaid Shahid**')
        st.divider()
        st.markdown('**The model listens for**')
        st.write(' · '.join(label.replace('_', ' ').title() for label in class_names))
        st.divider()
        st.markdown('**Model**\n\nFrozen YAMNet + trained 10-class head')
        st.caption('Mono audio · 16 kHz · first five seconds')
        st.link_button('Explore the ESC-10 dataset ↗', DATASET_URL, width='stretch')
        st.caption('The dashboard can be used without a viewer account.')

    first, second, third = st.columns(3)
    first.metric('Supported sounds', str(len(class_names)))
    second.metric('Listening window', '5 seconds')
    third.metric('Reported test accuracy', f"{metrics['test_accuracy']:.0%}" if 'test_accuracy' in metrics else 'Not supplied')
    st.caption('Reported accuracy is from 80 held-out ESC-10 clips on one fixed split. New recordings may perform differently.')

    classifier_tab, dataset_tab, evaluation_tab, resources_tab = st.tabs([
        '🎧 Sound classifier', '📚 Dataset & model', '📊 Evaluation', '📦 Downloads & links'])

    with classifier_tab:
        st.subheader('What are we listening to?')
        upload_column, result_column = st.columns([1, 1], gap='large')
        content, filename = None, None
        with upload_column, st.container(border=True):
            st.markdown('#### 01 · Add your audio')
            source = st.radio('Choose audio source', ['Upload WAV', 'Try a sample'],
                              horizontal=True, key='audio_source')
            if source == 'Upload WAV':
                uploaded = st.file_uploader('Choose a WAV recording', type=['wav'],
                                            max_upload_size=10, key='wav_upload',
                                            help='Up to 10 MB and 30 seconds. The model uses the first five seconds.')
                if uploaded is not None:
                    content, filename = uploaded.getvalue(), uploaded.name
            else:
                sample = st.selectbox('Pick an example sound', class_names,
                                      format_func=lambda value: value.replace('_', ' ').title(),
                                      key='sample_class')
                sample_path = ROOT / 'assets/samples' / (sample + '.wav')
                if sample_path.is_file():
                    content, filename = sample_path.read_bytes(), sample_path.name
                else:
                    st.warning('This sample is unavailable. Please upload a WAV recording instead.')
                st.caption('These examples are from the original ESC-10 test set.')

            audio_hash = hashlib.sha256(content).hexdigest() if content is not None else None
            if st.session_state.get('active_audio_hash') != audio_hash:
                st.session_state['active_audio_hash'] = audio_hash
                st.session_state.pop('prediction', None)
            waveform, audio_info = None, None
            if content is not None:
                try:
                    waveform, audio_info = preprocess_wav(content)
                    st.audio(content, format='audio/wav')
                    st.caption(f"{audio_info['original_duration']:.1f}s recording · "
                               f"{audio_info['original_sample_rate']:,} Hz · "
                               f"{audio_info['analysed_seconds']:.1f}s analysed")
                    chart_data = pd.DataFrame({'Time (s)': np.arange(0, SAMPLES, 160) / SAMPLE_RATE,
                                               'Amplitude': waveform[::160]}).set_index('Time (s)')
                    st.line_chart(chart_data, height=145, color='#5c8756', x_label='Time (seconds)',
                                  y_label='Amplitude')
                except ValueError as exc:
                    st.error(str(exc))
                    st.session_state.pop('prediction', None)
            else:
                st.info('Upload a WAV file or choose a sample to begin.')

            if st.button('Classify sound →', key='classify_sound', type='primary',
                         width='stretch', disabled=waveform is None):
                try:
                    with st.spinner('Listening to your sound…'):
                        result = classify(waveform, audio_info, class_names, model_sha256)
                        result['filename'] = filename
                        st.session_state['prediction'] = result
                except (ValueError, FileNotFoundError) as exc:
                    st.error(str(exc))
                    st.session_state.pop('prediction', None)
                except Exception:
                    st.error('The model could not complete this prediction. Please try again.')
                    st.session_state.pop('prediction', None)
            st.caption('WAV · up to 10 MB · first five seconds used. The app processes audio in memory and does not write your recording to disk.')

        with result_column, st.container(border=True):
            st.markdown('#### 02 · Meet your sound')
            result = st.session_state.get('prediction')
            if result is None:
                st.markdown('### 🎧 Ready to listen')
                st.write('Your predicted sound and scores for all ten classes will appear here.')
                st.caption('Choose a recording, then press Classify sound.')
            else:
                st.success('Analysis complete')
                st.subheader(result['display_label'])
                score_column, speed_column = st.columns(2)
                score_column.metric('Model score', f"{result['score']:.1%}")
                speed_column.metric('Processing time', f"{result['inference_ms'] / 1000:.2f}s")
                chart = pd.DataFrame([{'Sound': row['display_label'], 'Score (%)': row['score'] * 100}
                                      for row in result['scores']])
                st.bar_chart(chart, x='Sound', y='Score (%)', horizontal=True,
                             sort='-Score (%)', height=335, color='#5c8756',
                             x_label='Model score (%)', y_label='Sound class')
                if result['audio']['rms'] < 0.001:
                    st.warning('This recording is very quiet. Try a clearer example of one supported sound.')
                elif result['score'] < 0.6:
                    st.warning('The model score is low. Try a clearer recording of a single supported sound.')
                st.caption('Scores are estimates across ten known sounds. Unfamiliar or mixed sounds can be misclassified; scores are not calibrated probabilities of correctness.')
                st.download_button('Download prediction JSON', json.dumps(result, indent=2),
                                   file_name='soundsense_prediction.json', mime='application/json',
                                   key='download_prediction', width='stretch')

    with dataset_tab:
        st.subheader('One model. Ten familiar sounds.')
        st.write('The project uses the **ESC-10 subset**: 400 five-second recordings, with 40 recordings in each class.')
        st.dataframe(pd.DataFrame({'Class': [name.replace('_', ' ').title() for name in class_names],
                                   'Train': [24] * 10, 'Validation': [8] * 10, 'Test': [8] * 10}),
                     hide_index=True, width='stretch')
        st.markdown('**Split:** original folds 3–5 train the classifier (240 clips), fold 2 validates it (80 clips), and fold 1 tests it (80 clips).')
        st.markdown('**Architecture:** YAMNet extracts 1,024-dimensional embeddings; mean pooling is followed by trained normalization, Dense128 ReLU, Dropout0.35 and Dense10 softmax.')
        st.markdown('**Audio pipeline:** decode WAV → average channels → resample to 16 kHz → keep the first five seconds → clip to −1…1 → zero-pad short clips.')
        st.info('The notebook trains the classifier. The saved model performs predictions in this dashboard.')
        st.link_button('Dataset source and documentation ↗', DATASET_URL)

    with evaluation_tab:
        st.subheader('Measured model performance')
        if metrics:
            a, b, c = st.columns(3)
            a.metric('Correct test predictions', f"{metrics.get('test_correct', '—')} / {metrics.get('test_clips', '—')}")
            f1 = metrics.get('classification_report', {}).get('macro avg', {}).get('f1-score')
            b.metric('Macro F1', f'{f1:.2f}' if f1 is not None else '—')
            c.metric('Selected epoch', f"{metrics.get('selected_epoch', '—')} / {metrics.get('completed_epochs', '—')}")
        st.caption('One fixed fold-based holdout, not a five-fold cross-validation average or a guarantee on new recordings.')
        left, right = st.columns(2)
        if (ROOT / 'results/confusion_matrix.png').is_file():
            left.image(str(ROOT / 'results/confusion_matrix.png'), caption='Original test confusion matrix', width='stretch')
        if (ROOT / 'results/training_curves.png').is_file():
            right.image(str(ROOT / 'results/training_curves.png'), caption='Original training and validation history', width='stretch')
        verification = ROOT / 'results/conversion_verification.json'
        if verification.is_file():
            with st.expander('TensorFlow / ONNX conversion verification'):
                st.json(read_json(verification))

    with resources_tab:
        st.subheader('Everything behind the sound')
        left, right = st.columns(2)
        with left:
            download_file('Download complete trained model (.onnx)', 'model/soundsense.onnx',
                          mime='application/octet-stream', key='download_model')
            download_file('Download model specification (.json)', 'model/manifest.json',
                          mime='application/json', key='download_manifest')
            download_file('Download class order (.json)', 'model/class_names.json',
                          mime='application/json', key='download_classes')
        with right:
            download_file('Download training notebook (.ipynb)', 'notebooks/SoundSense_Training.ipynb',
                          mime='application/json', key='download_notebook')
            download_file('Download original test predictions (.csv)', 'results/test_predictions.csv',
                          mime='text/csv', key='download_test_predictions')
            download_file('Download dataset split manifest (.csv)', 'results/dataset_manifest.csv',
                          mime='text/csv', key='download_dataset_manifest')
        st.markdown('**Project submission links**')
        names = {'github': 'GitHub repository', 'report': 'Project report', 'presentation': 'Presentation',
                 'youtube': 'YouTube presentation', 'linkedin': 'LinkedIn post'}
        link_columns = st.columns(3)
        for index, (key, label) in enumerate(names.items()):
            url = links.get(key)
            valid = isinstance(url, str) and urlparse(url).scheme == 'https' and bool(urlparse(url).netloc)
            if valid:
                link_columns[index % 3].link_button(label + ' ↗', url, width='stretch')
            else:
                link_columns[index % 3].caption(label + ' · Not added yet')
        with st.expander('Dataset attribution and model license'):
            st.write('ESC-10: Karol J. Piczak, 2015. The ESC-10 subset is CC BY; included files retain the clip attributions. YAMNet comes from TensorFlow Models under Apache 2.0.')
            download_file('ESC-10 source attributions', 'licenses/ESC10_attributions.txt',
                          mime='text/plain', key='download_dataset_license')
            download_file('YAMNet license', 'licenses/YAMNET_LICENSE.txt',
                          mime='text/plain', key='download_model_license')
        st.caption('This app uses the complete trained model converted to ONNX. The classification head alone cannot process raw audio.')

    st.divider()
    st.caption('SoundSense · Environmental Sound Classification Using Deep Learning · Junaid Shahid')


if __name__ == '__main__':
    main()
