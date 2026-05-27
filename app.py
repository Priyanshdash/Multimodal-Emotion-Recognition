import os
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple, Dict

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["ABSL_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HUGGINGFACE_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_CACHE"] = str(Path(__file__).resolve().parents[0] / ".hf_cache")
os.environ["HF_HOME"] = str(Path(__file__).resolve().parents[0] / ".hf_cache")
os.environ["HF_HUB_CACHE"] = str(Path(__file__).resolve().parents[0] / ".hf_cache" / "hub")

import gradio as gr
import librosa
import logging
import numpy as np
import tensorflow as tf
import torch
from transformers import (
    AutoFeatureExtractor,
    BertModel,
    BertTokenizer,
    HubertModel,
    logging as transformers_logging,
)

PROJECT_ROOT = Path(__file__).resolve().parents[0]
CACHE_DIR = PROJECT_ROOT / ".hf_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
transformers_logging.set_verbosity_error()
transformers_logging.enable_default_handler()
transformers_logging.disable_propagation()
logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_SEARCH_ROOTS = [PROJECT_ROOT, MODELS_DIR, PROJECT_ROOT.parent]


def find_model_file(filename: str) -> Optional[Path]:
    for root in MODEL_SEARCH_ROOTS:
        if root is None or not root.exists():
            continue
        matches = list(root.rglob(filename))
        if matches:
            return matches[0]
    return None


SPEECH_MODEL_PATH = MODELS_DIR / "speech_pipeline" / "speech_emotion_model.h5"
if not SPEECH_MODEL_PATH.exists():
    SPEECH_MODEL_PATH = find_model_file("speech_emotion_model.h5")

TEXT_MODEL_PATH = MODELS_DIR / "text_pipeline" / "text_emotion_model.h5"
if not TEXT_MODEL_PATH.exists():
    TEXT_MODEL_PATH = find_model_file("text_emotion_model.h5")

FUSION_MODEL_PATH = MODELS_DIR / "fusion_pipeline" / "fusion_emotion_model.h5"
if not FUSION_MODEL_PATH.exists():
    FUSION_MODEL_PATH = find_model_file("fusion_emotion_model.h5")

SPEECH_LABEL_ENCODER_PATH = MODELS_DIR / "speech_pipeline" / "label_encoder.pkl"
if not SPEECH_LABEL_ENCODER_PATH.exists():
    SPEECH_LABEL_ENCODER_PATH = find_model_file("label_encoder.pkl")

TEXT_LABEL_ENCODER_PATH = MODELS_DIR / "text_pipeline" / "text_label_encoder.pkl"
if not TEXT_LABEL_ENCODER_PATH.exists():
    TEXT_LABEL_ENCODER_PATH = find_model_file("text_label_encoder.pkl")

FUSION_LABEL_ENCODER_PATH = SPEECH_LABEL_ENCODER_PATH or TEXT_LABEL_ENCODER_PATH


def load_label_encoder(path: Optional[Path]):
    if path is None or not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def get_speech_model():
    if SPEECH_MODEL_PATH is None or not SPEECH_MODEL_PATH.exists():
        raise FileNotFoundError("Speech model file not found. Expected speech_emotion_model.h5")
    return tf.keras.models.load_model(str(SPEECH_MODEL_PATH), compile=False)


@lru_cache(maxsize=1)
def get_text_model():
    if TEXT_MODEL_PATH is None or not TEXT_MODEL_PATH.exists():
        raise FileNotFoundError("Text model file not found. Expected text_emotion_model.h5")
    return tf.keras.models.load_model(str(TEXT_MODEL_PATH), compile=False)


@lru_cache(maxsize=1)
def get_fusion_model():
    if FUSION_MODEL_PATH is None or not FUSION_MODEL_PATH.exists():
        raise FileNotFoundError("Fusion model file not found. Expected fusion_emotion_model.h5")
    return tf.keras.models.load_model(str(FUSION_MODEL_PATH), compile=False)


@lru_cache(maxsize=1)
def get_speech_feature_extractor():
    return AutoFeatureExtractor.from_pretrained("facebook/hubert-base-ls960")


@lru_cache(maxsize=1)
def get_hubert_model():
    model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(DEVICE)
    model.eval()
    return model


@lru_cache(maxsize=1)
def get_text_tokenizer():
    return BertTokenizer.from_pretrained('bert-base-uncased')


@lru_cache(maxsize=1)
def get_bert_model():
    model = BertModel.from_pretrained('bert-base-uncased').to(DEVICE)
    model.eval()
    return model


@lru_cache(maxsize=1)
def get_speech_label_encoder():
    encoder = load_label_encoder(SPEECH_LABEL_ENCODER_PATH)
    if encoder is None:
        raise FileNotFoundError("Speech label encoder not found. Expected label_encoder.pkl")
    return encoder


@lru_cache(maxsize=1)
def get_text_label_encoder():
    encoder = load_label_encoder(TEXT_LABEL_ENCODER_PATH)
    if encoder is None:
        raise FileNotFoundError("Text label encoder not found. Expected text_label_encoder.pkl")
    return encoder


@lru_cache(maxsize=1)
def get_fusion_label_encoder():
    encoder = load_label_encoder(FUSION_LABEL_ENCODER_PATH)
    if encoder is None:
        raise FileNotFoundError("Fusion label encoder not found. Expected label_encoder.pkl or text_label_encoder.pkl")
    return encoder


def load_audio(audio_input) -> Optional[np.ndarray]:
    if audio_input is None:
        return None

    if isinstance(audio_input, str):
        audio, sr = librosa.load(audio_input, sr=16000)
        return audio.astype(np.float32)

    if isinstance(audio_input, tuple) and len(audio_input) == 2:
        audio, sr = audio_input
    else:
        audio = np.asarray(audio_input)
        sr = 16000

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    return audio.astype(np.float32)


def extract_speech_embedding(audio: np.ndarray) -> np.ndarray:
    inputs = get_speech_feature_extractor()(audio, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = get_hubert_model()(**inputs)
    hidden_states = outputs.last_hidden_state
    return hidden_states.mean(dim=1).cpu().numpy().squeeze()


def extract_text_embedding(text: str) -> np.ndarray:
    encoded = get_text_tokenizer()(text, truncation=True, padding='max_length', max_length=64, return_tensors='pt')
    encoded = {k: v.to(DEVICE) for k, v in encoded.items()}
    with torch.no_grad():
        outputs = get_bert_model()(**encoded)
    return outputs.pooler_output.cpu().numpy().squeeze()


def build_label_output(predictions: np.ndarray, label_encoder) -> Tuple[str, Dict[str, float]]:
    probabilities = predictions.squeeze().tolist()
    label_idx = int(np.argmax(probabilities, axis=-1))
    emotion_label = label_encoder.inverse_transform([label_idx])[0]
    return emotion_label, {
        label_encoder.inverse_transform([i])[0]: float(probabilities[i])
        for i in range(len(probabilities))
    }


def predict_emotion(model_choice, speech_audio, text_input):
    audio_input = speech_audio
    text_input = text_input.strip() if isinstance(text_input, str) else ''

    try:
        if model_choice == 'speech':
            if audio_input is None:
                return 'Please record or upload a WAV audio file.', {}, 'Speech input is required.'

            audio = load_audio(audio_input)
            embedding = extract_speech_embedding(audio)
            predictions = get_speech_model().predict(np.expand_dims(embedding, axis=0), verbose=0)
            label, probabilities = build_label_output(predictions, get_speech_label_encoder())
            return label, probabilities, 'Speech-only model prediction.'

        if model_choice == 'text':
            if not text_input:
                return 'Please type a text input.', {}, 'Text input is required.'

            embedding = extract_text_embedding(text_input)
            predictions = get_text_model().predict(np.expand_dims(embedding, axis=0), verbose=0)
            label, probabilities = build_label_output(predictions, get_text_label_encoder())
            return label, probabilities, 'Text-only model prediction.'

        if model_choice == 'fusion':
            if audio_input is None or not text_input:
                return 'Please provide both speech and text for the fusion model.', {}, 'Both speech and text inputs are required for fusion.'

            speech_emb = extract_speech_embedding(load_audio(audio_input))
            text_emb = extract_text_embedding(text_input)
            fusion_input = np.expand_dims(np.concatenate([speech_emb, text_emb], axis=-1), axis=0)
            predictions = get_fusion_model().predict(fusion_input, verbose=0)
            label, probabilities = build_label_output(predictions, get_fusion_label_encoder())
            return label, probabilities, 'Fusion model prediction using both speech and text.'

        return 'Unknown model selection.', {}, 'Please select speech, text, or fusion.'
    except Exception as exc:
        return 'Prediction error', {}, str(exc)


def update_input_visibility(model_choice):
    speech_visible = model_choice in ['speech', 'fusion']
    text_visible = model_choice in ['text', 'fusion']
    return (
        gr.update(visible=speech_visible),
        gr.update(visible=text_visible),
    )


def main():
    with gr.Blocks(title='Multimodal Emotion Recognition') as demo:
        gr.Markdown(
            '# Multimodal Emotion Recognition UI\n'
            'Use the toggle below to choose between Speech, Text, and Fusion models.\n'
            '- Speech: record your voice or upload a WAV file. If the browser does not detect a microphone, please use the upload option instead.\n'
            '- Text: type a sentence.\n'
            '- Fusion: provide both speech and text together.\n'
        )

        model_choice = gr.Radio(
            ['speech', 'text', 'fusion'],
            label='Select model',
            value='speech'
        )

        with gr.Row():
            with gr.Column():
                speech_audio = gr.Audio(
                    sources=['microphone', 'upload'],
                    type='filepath',
                    label='Speech input (record or upload)',
                    recording=False,
                    visible=True,
                )
            with gr.Column():
                text_input = gr.Textbox(
                    label='Type text here',
                    placeholder='Enter text for text-only or fusion model',
                    lines=4,
                    visible=False,
                )

        predict_button = gr.Button('Classify')
        output_label = gr.Textbox(label='Predicted emotion', interactive=False)
        output_probs = gr.Label(label='Emotion probabilities')
        status = gr.Textbox(label='Status', interactive=False)

        model_choice.change(
            update_input_visibility,
            inputs=model_choice,
            outputs=[speech_audio, text_input],
        )

        predict_button.click(
            predict_emotion,
            inputs=[model_choice, speech_audio, text_input],
            outputs=[output_label, output_probs, status],
        )

        gr.Markdown('**Instructions:** For speech-only, use either record or upload. For fusion, provide both speech and text.')

    demo.launch()


def warm_up_models_async():
    """Warm up tokenizers/feature-extractors/models in background to avoid UI-triggered downloads."""
    try:
        get_text_tokenizer()
        get_speech_feature_extractor()
        get_text_model()
        get_speech_model()
        get_fusion_model()
        print("Background model warm-up completed.")
    except Exception as e:
        print("Background warm-up error:", e)


if __name__ == '__main__':
    import threading
    threading.Thread(target=warm_up_models_async, daemon=True).start()
    main()
