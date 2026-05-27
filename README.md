# Multimodal Emotion Recognition

This project detects emotions from speech, text, and combined speech + text
inputs using a multimodal deep learning pipeline. It provides trained models,
training notebooks, evaluation artifacts, and a Gradio app for interactive
testing.

## What This Project Does

- Speech emotion classification using HuBERT embeddings and a Keras model
- Text emotion classification using BERT embeddings and a Keras model
- Fusion-based emotion classification using concatenated speech and text
  embeddings
- Interactive inference using a Gradio web interface

## Architecture Diagrams

### Complete Pipeline

![Complete Architecture](Architectures/Complete_architecture.jpg)

### Speech Pipeline

![Speech Architecture](Architectures/Speech_architecture.jpg)

### Text Pipeline

![Text Architecture](Architectures/Text_architecture.jpg)

### Fusion Pipeline

![Fusion Architecture](Architectures/Fusion_architecture.jpg)

## Model Architectures

### Speech Model

Notebook: models/speech_pipeline/speechv3.ipynb

- Input: HuBERT mean pooled embedding (768)
- Reshape to (1, 768)
- Bidirectional LSTM (128)
- Dropout (0.3)
- Dense (128, ReLU)
- Dropout (0.3)
- Dense (num classes, Softmax)

### Text Model

Notebook: models/text_pipeline/textv3.ipynb

- Input: BERT pooler output (768)
- Dense (256, ReLU)
- Dropout (0.3)
- Dense (128, ReLU)
- Dropout (0.3)
- Dense (num classes, Softmax)

### Fusion Model

Notebook: models/fusion_pipeline/fusionv3.ipynb

- Input: Concatenated speech + text embeddings (1536)
- Dense (512, ReLU)
- Dropout (0.4)
- Dense (256, ReLU)
- Dropout (0.4)
- Dense (128, ReLU)
- Dropout (0.3)
- Dense (num classes, Softmax)

## Results

### Metrics Files

- Results/metrics/speech.csv
- Results/metrics/text.csv
- Results/metrics/fusion.csv

### Summary Metrics

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| Speech | 0.996429 | 0.996473 | 0.996429 | 0.996417 |
| Text | 0.140000 | 0.020000 | 0.140000 | 0.040000 |
| Fusion | 0.992857 | 0.992988 | 0.992857 | 0.992856 |

### Confusion Matrices

![Speech Confusion Matrix](Results/confusion_matrices/speech_comfusion_matrix.png)

![Text Confusion Matrix](Results/confusion_matrices/text_confusion_matrix.png)

![Fusion Confusion Matrix](Results/confusion_matrices/fusion_confusion_matrix.png)

### Training Curves

Speech:

- ![Speech Accuracy](Results/plots/speech/speech_training.png)
- ![Speech Loss](Results/plots/speech/speech_loss.png)

Text:

- ![Text Accuracy](Results/plots/text/text_accuracy.png)
- ![Text Loss](Results/plots/text/text_loss.png)

Fusion:

- ![Fusion Accuracy](Results/plots/fusion/fusion_accuracy.png)
- ![Fusion Loss](Results/plots/fusion/fusion_loss.png)

## Project Structure

- app.py: Gradio application and inference logic
- requirements.txt: Python dependencies
- models/: trained checkpoints and model artifacts
- data/TESS/: dataset folders used for training and evaluation
- Results/: metrics, confusion matrices, and plots
- MultiModal_Emotion_Recognition_Report.pdf: full project report

## How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the app:

```bash
python app.py
```

3. Open the Gradio URL shown in the terminal.

## Checkpoint Files Required

- models/speech_pipeline/speech_emotion_model.h5
- models/text_pipeline/text_emotion_model.h5
- models/fusion_pipeline/fusion_emotion_model.h5

## Notes

- Use Git LFS for large model files.
- Hugging Face models are downloaded on first run and cached locally.
- For reproducibility, pin exact versions in requirements.txt.

## Attribution

Please cite the TESS dataset and pretrained Hugging Face models used in this
project if you reuse this work.

