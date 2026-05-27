# Multimodal Emotion Recognition

This repository contains a Gradio demo and supporting artifacts for a multimodal
emotion recognition system that predicts emotions from speech, text, or a
combination (fusion) of both.

The app uses pretrained Keras `.h5` models for classification and Hugging Face
Transformers models (HuBERT for audio features and BERT for text features) to
produce embeddings during inference.

## Key Features

- Speech-only emotion recognition (HuBERT embeddings → Keras classifier)
- Text-only emotion recognition (BERT embeddings → Keras classifier)
- Fusion model combining speech + text embeddings for improved accuracy
- Interactive demo powered by Gradio (`app.py`)

## Quick Start

Prerequisites: Python 3.9+, `pip`, and (optionally) `git lfs` if you keep model
checkpoints in the repository.

Install dependencies:

```bash
cd project
pip install -r requirements.txt
```

Run the demo:

```bash
python app.py
```

Open the local URL printed by Gradio in your browser to use the UI.

## Using the App

The UI exposes three modes via a radio control:

- `speech`: record or upload a WAV file for speech-only prediction.
- `text`: type a sentence for text-only prediction.
- `fusion`: provide both a speech file and text input; the fusion model will be
    used.

The app returns a predicted emotion label and a probability distribution over
all supported emotions.

## Repository Layout

- `app.py` — Gradio application and model-loading/prediction code.
- `requirements.txt` — Python dependencies for running the demo.
- `data/TESS/` — expected location for the TESS audio dataset used for
    training/evaluation (not required to run the demo if models exist).
- `models/` — pretrained model checkpoints and auxiliary files for each
    pipeline (`speech_pipeline`, `text_pipeline`, `fusion_pipeline`).
- `Results/` — evaluation outputs: confusion matrices, metrics and plots.
- `*.ipynb` — notebooks used for training/analysis (kept for reproducibility).

## Models and Checkpoints

The demo expects Keras `.h5` checkpoint files in the following locations:

- `models/speech_pipeline/speech_emotion_model.h5`
- `models/text_pipeline/text_emotion_model.h5`
- `models/fusion_pipeline/fusion_emotion_model.h5`

Label encoders and small numpy artifacts (e.g. `label_encoder.pkl`, `*.npy`)
should remain with the model folders. Large binary checkpoints are often better
hosted outside the repository (or stored with Git LFS).

## Evaluation & Results

Evaluation artifacts (confusion matrices, loss/accuracy plots) are stored in
`Results/`. The notebooks show how the training and metrics were produced.

### Results & CSVs

- Metrics CSVs: `Results/metrics/` contains CSV summaries of evaluation
    metrics (e.g. `speech_metrics.csv`, `text_metrics.csv`, `fusion_metrics.csv`).
- Confusion matrices: `Results/confusion_matrices/` contains saved matrices
    (PNG/CSV) for each pipeline.
- Plots: `Results/plots/` includes training curves and visual summaries used
    in the report.

If you want to include the CSVs in a paper or report, open the files in
`Results/metrics/` and extract the rows for precision/recall/F1/accuracy per
class. The notebooks show exact code used to generate those CSVs.

### Example (how to read a CSV)

Use pandas to load the fusion metrics and view results quickly:

```python
import pandas as pd
df = pd.read_csv('Results/metrics/fusion_metrics.csv')
print(df)
```

## Architectures (brief)

- Speech pipeline: pretrained HuBERT encoder → mean-pooling of last hidden
    states → small dense classifier (Keras) trained on TESS-derived features.
- Text pipeline: `bert-base-uncased` tokenizer + BERT model → pooler output →
    dense classifier (Keras).
- Fusion pipeline: concatenation of speech and text embeddings → fully
    connected layers → softmax classifier. The fusion model typically adds one
    or two dense layers (dropout + ReLU) on top of concatenated embeddings.

For full architecture details and layer sizes, see the training notebooks
(`models/*/` notebooks) which include model summaries and the code used to
construct each Keras model.

### Exact model architectures (as used in notebooks)

- Speech model (`models/speech_pipeline/speechv3.ipynb`):

    - Input: HuBERT mean-pooled embedding (shape: 768)
    - Reshape to (1, 768)
    - Bidirectional LSTM (128 units, return_sequences=False)
    - Dropout (0.3)
    - Dense (128, ReLU)
    - Dropout (0.3)
    - Dense (num_classes, softmax)

- Text model (`models/text_pipeline/textv3.ipynb`):

    - Input: BERT pooler output (shape: 768)
    - Dense (256, ReLU)
    - Dropout (0.3)
    - Dense (128, ReLU)
    - Dropout (0.3)
    - Dense (num_classes, softmax)

- Fusion model (`models/fusion_pipeline/fusionv3.ipynb`):

    - Input: Concatenated [speech_emb (768) + text_emb (768)] → shape 1536
    - Dense (512, ReLU)
    - Dropout (0.4)
    - Dense (256, ReLU)
    - Dropout (0.4)
    - Dense (128, ReLU)
    - Dropout (0.3)
    - Dense (num_classes, softmax)

These exact layer definitions are in the notebooks and were used to produce
the checkpoints saved under `models/*_pipeline/`.

## Results (detailed files)

The project includes evaluation spreadsheets and plots generated during
training. You can find the numeric results here:

- `Results/metrics/fusion.xlsx` — fusion evaluation metrics (per-class and
    overall precision/recall/F1/accuracy)
- `Results/metrics/speech.xlsx` — speech-only metrics
- `Results/metrics/text.xlsx` — text-only metrics
- `Results/confusion_matrices/` — confusion matrix images and CSVs
- `Results/plots/` — training curves (loss/accuracy) and other visualizations

To convert the `.xlsx` files to CSV for quick inspection:

```python
import pandas as pd
pd.read_excel('Results/metrics/fusion.xlsx').to_csv('Results/metrics/fusion.csv', index=False)
```

Open the resulting CSV with Excel, pandas, or any spreadsheet viewer. If you
want, I can extract the key metrics (accuracy, macro F1) from the `.xlsx`
files and add a small summary table into this README.

## Small Project Summary

This project implements a multimodal emotion recognition system that
combines speech and text inputs to improve classification accuracy over
single-modality baselines. It includes pretrained checkpoints, evaluation
artifacts, and interactive demo via Gradio. The goal is to provide a
reproducible pipeline for exploring multimodal fusion for emotion tasks.


## Reproducibility notes

- Pin dependency versions in `requirements.txt` for exact reproducibility.
- If you keep `.h5` model files in the repo, use Git LFS to avoid hitting
    GitHub file-size limits.
- The app downloads Hugging Face models at runtime unless cached in
    `.hf_cache/`.

## Troubleshooting

- If the app fails with a missing-model error: confirm the `.h5` files exist at
    the paths above or update `app.py` to point to the correct files.
- If Hugging Face model downloads fail, ensure you have network access and
    sufficient disk space; set `TRANSFORMERS_CACHE`/`HF_HOME` environment
    variables to control cache location.

## Contributing

Contributions are welcome — open an issue or submit a pull request with a
clear description of changes. For large files, provide download links rather
than committing binaries directly.

## License & Attribution

Include a `LICENSE` file for this repository if you wish to make the license
explicit. Cite the TESS dataset and any pretrained models used when publishing
results derived from this work.

