import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import json
from preprocess import preprocess_text

class ModelService:
    def __init__(self, model_path="./models/sentiment_model"):
        self.model_path = model_path
        if not os.path.exists(model_path) or not os.listdir(model_path):
            self.model = None
            self.tokenizer = None
            self.label_mapping = None
            print(f"Warning: Model path {model_path} is empty or does not exist.")
            return

        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)

        # Move model to available device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        mapping_path = os.path.join(model_path, "label_mapping.json")
        if os.path.exists(mapping_path):
            with open(mapping_path, "r", encoding='utf-8') as f:
                self.label_mapping = json.load(f)
        else:
            self.label_mapping = None
        print(f"Model loaded successfully on {self.device}.")

    def predict(self, text: str):
        if self.model is None or self.tokenizer is None:
            return {"error": "Model not loaded. Please train the model first."}

        processed_text = preprocess_text(text)
        inputs = self.tokenizer(processed_text, return_tensors="pt", truncation=True, max_length=256, padding=True)

        # Move inputs to same device as model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)

        probs = probabilities[0].cpu().tolist()

        results = []
        for i, prob in enumerate(probs):
            # Check for label in multiple places for robustness
            label = "unknown"
            if hasattr(self.model.config, 'id2label'):
                label = self.model.config.id2label.get(str(i), self.model.config.id2label.get(i, str(i)))
            elif self.label_mapping and 'id2label' in self.label_mapping:
                label = self.label_mapping['id2label'].get(str(i), self.label_mapping['id2label'].get(i, str(i)))

            results.append({"label": label, "confidence": float(prob)})

        # Sort by confidence descending
        results = sorted(results, key=lambda x: x["confidence"], reverse=True)

        return {
            "text": text,
            "processed_text": processed_text,
            "predictions": results,
            "top_prediction": results[0]
        }

# Singleton instance
service = None

def get_model_service(force_reload=False):
    global service
    if service is None or force_reload:
        service = ModelService()
    return service
