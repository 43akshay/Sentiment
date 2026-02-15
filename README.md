# SentiMind: Mental Health & Sentiment AI

SentiMind is a comprehensive system designed to analyze both emotional sentiment and mental health indicators from text. Built on DistilBERT, it uses multi-label classification to provide a nuanced understanding of a user's wellbeing.

## Project Structure

- `train.py`: Fine-tunes the DistilBERT model on labeled datasets (supports multi-label).
- `model_service.py`: Central service for loading the model and performing complex inference.
- `api.py`: FastAPI server providing a `/predict` endpoint for integrations.
- `ui/app.py`: Modern Streamlit dashboard for interactive analysis.
- `predict.py`: Command-line tool for quick predictions.
- `preprocess.py`: Text cleaning and normalization module.
- `data/`: Directory containing training datasets (`train.txt`, `mental.txt`).

## Installation

1. Clone the repository and navigate to the root directory.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Training the Model

The system allows you to train on multiple datasets simultaneously. By default, it uses both the general sentiment data and the specialized mental health data.

To train with default settings:
```bash
python train.py
```

To customize training (e.g., change epochs or data files):
```bash
python train.py --data data/train.txt,data/mental.txt --epochs 5 --batch_size 16
```

**Dataset Format:**
Data should be in a semicolon-separated `.txt` file: `text;label1,label2`.
*Example:* `I feel so overwhelmed and sad;sadness,stress`

## Usage

### 1. Web Dashboard (Recommended)
Start the polished SentiMind UI to get visual reports and history tracking:
```bash
streamlit run ui/app.py
```

### 2. CLI Prediction
Analyze text directly from your terminal:
```bash
python predict.py "I've been feeling a bit anxious lately about work."
```
Use the `--json` flag for machine-readable output.

### 3. FastAPI Server
Serve predictions via a REST API:
```bash
python api.py
```
Send a POST request to `http://localhost:8000/predict`:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "I am proud of my progress in therapy."}'
```

## How It Works

- **Overall Sentiment**: Calculated on a scale of -1 to 1 based on positive (joy, love) vs negative (anger, fear, sadness, etc.) emotional traits.
- **Mental Health Indicator**: Specifically identifies patterns associated with Anxiety, Depression, Stress, or Bipolar states.
- **Multi-Label Logic**: The system recognizes that emotions are rarely singular; it can detect multiple overlapping states in a single piece of text.

---
*Disclaimer: This tool is for informational purposes only and is not a substitute for professional medical advice or diagnosis.*
