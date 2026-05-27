# Multimodal Emotion Recognition

A GitHub-ready repository for a Gradio-based emotion recognition app using speech, text, and multimodal fusion models.

The app loads pretrained Keras `.h5` models and uses Hugging Face Transformers to compute speech and text embeddings during prediction.

## Features

- Speech emotion recognition using HuBERT-based audio embeddings.
- Text emotion recognition using BERT-based text embeddings.
- Fusion emotion recognition using combined speech and text representations.
- Live UI powered by Gradio.

## Repository Contents

- `app.py` — main application launching the Gradio UI.
- `requirements.txt` — Python dependencies.
- `models/` — pretrained model checkpoints and notebook artifacts for each pipeline.
- `data/TESS/` — expected location for the TESS dataset audio files.
- `Results/` — evaluation outputs, confusion matrices, and plots.

## Available checkpoints

- `models/speech_pipeline/text_emotion_model.h5`
- `models/text_pipeline/speech_emotion_model.h5`
- `models/fusion_pipeline/fusion_emotion_model.h5`

> These checkpoint files are loaded automatically by `app.py`.

## Folder structure

```text
project/
├── .hf_cache/
├── app.py
├── data/
│   └── TESS/
├── models/
│   ├── fusion_pipeline/
│   │   ├── fusion_emotion_model.h5
│   │   └── fusionv3.ipynb
│   ├── speech_pipeline/
│   │   ├── speechv3.ipynb
│   │   ├── text_emotion_model.h5
│   │   ├── bert_features.npy
│   │   └── text_labels.npy
│   └── text_pipeline/
│       ├── textv3.ipynb
│       ├── speech_emotion_model.h5
│       ├── emotion_labels.npy
│       └── hubert_features.npy
├── README.md
├── requirements.txt
└── Results/
    ├── confusion_matrices/
    ├── fusion/
    ├── metrics/
    ├── plots/
    ├── speech/
    └── text/
```

## Getting Started

### Prerequisites

- Python 3.9+ (or compatible version)
- `pip`

### Installation

```bash
cd project
pip install -r requirements.txt
```

### Run the application

```bash
python app.py
```

Open the local URL shown in the terminal to access the Gradio interface.

## Usage

The UI supports three modes:

- `speech` — upload or record a WAV audio file.
- `text` — enter a text sentence.
- `fusion` — provide both speech and text for combined prediction.

## Notes for GitHub

- This repository is ready to push as a GitHub project.
- Keep the `.h5` model checkpoint filenames the same if you replace them.
- The notebooks are included for reference and are not required to run the app.

## Replacing model checkpoints

To swap in new models, copy the new `.h5` checkpoint into the appropriate folder:

- `models/speech_pipeline/`
- `models/text_pipeline/`
- `models/fusion_pipeline/`

Then restart `app.py`.

## Troubleshooting

- If models fail to load, verify the checkpoint files exist in the expected folders.
- TensorFlow and Hugging Face warnings are usually informational and do not prevent the app from running.
- If import or dependency errors occur, use a clean Python environment and reinstall dependencies.
