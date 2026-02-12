# Custom Sentiment Analysis System

This system allows you to train a custom sentiment analysis model using DistilBERT on your own labeled dataset. It supports multi-label probability outputs to capture complex "over the bridge" emotions.

## Project Structure

- `train.py`: Fine-tunes the model on a labeled `.txt` dataset.
- `predict.py`: CLI tool for sentiment prediction.
- `api.py`: FastAPI server providing a `/predict` endpoint.
- `model_service.py`: Core logic for loading the model and running inference.
- `preprocess.py`: Text preprocessing module.
- `ui/app.py`: Streamlit-based web interface.
- `data/`: Directory for training data.
- `models/`: Directory where the trained model is saved.
- `outputs/`: Directory for training logs and metrics.

## Installation

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Training

To train the model on the default dataset (`data/train.txt`):
```bash
python train.py
```

To train on a custom file:
```bash
python train.py --data path/to/your_data.txt --epochs 5
```

The dataset should be a `.txt` file with lines in the format: `text;label`.

## Usage

### CLI Prediction
```bash
python predict.py "I feel so amazing today!"
```

### FastAPI Server
Start the server:
```bash
python api.py
```
The API will be available at `http://localhost:8000`. You can send a POST request to `/predict`:
```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"text": "I am feeling a bit uncertain."}'
```

### Streamlit UI
Start the UI:
```bash
streamlit run ui/app.py
```

## Dataset Format
The system expects a semicolon-separated `.txt` file:
```
I love this;joy
I am so angry;anger
It is okay;neutral
```
The system automatically detects all unique labels and configures the model accordingly.
