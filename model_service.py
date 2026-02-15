import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import json
import numpy as np
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

        self.mental_health_labels = {'anxiety', 'bipolar', 'depression', 'stress'}
        self.sentiment_labels = {'joy', 'love', 'sadness', 'anger', 'fear', 'surprise', 'positive', 'neutral'}

    def predict(self, text: str):
        if self.model is None or self.tokenizer is None:
            return {"error": "Model not loaded. Please train the model first."}

        processed_text = preprocess_text(text)
        inputs = self.tokenizer(processed_text, return_tensors="pt", truncation=True, max_length=64, padding=True)

        # Move inputs to same device as model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            # For multi-label, use Sigmoid
            probabilities = torch.sigmoid(logits)

        probs = probabilities[0].cpu().tolist()

        results = []
        for i, prob in enumerate(probs):
            label = "unknown"
            if hasattr(self.model.config, 'id2label'):
                label = self.model.config.id2label.get(str(i), self.model.config.id2label.get(i, str(i)))
            elif self.label_mapping and 'id2label' in self.label_mapping:
                label = self.label_mapping['id2label'].get(str(i), self.label_mapping['id2label'].get(i, str(i)))

            results.append({"label": label, "confidence": float(prob)})

        # Sort by confidence descending
        results = sorted(results, key=lambda x: x["confidence"], reverse=True)

        # Split results into Sentiment and Mental Health
        sentiment_results = [r for r in results if r["label"] in self.sentiment_labels]
        mental_health_results = [r for r in results if r["label"] in self.mental_health_labels]

        prob_dict = {res["label"]: res["confidence"] for res in results}

        # Calculate Overall Sentiment
        pos_score = prob_dict.get("joy", 0) + prob_dict.get("love", 0) + prob_dict.get("positive", 0)
        # Mental health labels contribute to negative sentiment score
        neg_score = (prob_dict.get("anger", 0) + prob_dict.get("fear", 0) + prob_dict.get("sadness", 0) +
                     prob_dict.get("depression", 0) + prob_dict.get("anxiety", 0) +
                     prob_dict.get("stress", 0) + prob_dict.get("bipolar", 0) * 0.5)

        sentiment_score = pos_score - neg_score

        if sentiment_score > 0.1:
            overall_sentiment = "Positive"
        elif sentiment_score < -0.1:
            overall_sentiment = "Negative"
        else:
            overall_sentiment = "Neutral"

        # Mental Health Indicator
        top_mh = mental_health_results[0] if mental_health_results else {"label": "none", "confidence": 0}
        if top_mh["confidence"] > 0.4:
            mh_indicator = top_mh["label"].capitalize()
            mh_alert = True
        else:
            mh_indicator = "Stable"
            mh_alert = False

        # Key Logic
        top_label = results[0]["label"]
        logic_templates = {
            "joy": "The text expresses happiness and positivity.",
            "love": "The text conveys deep affection or appreciation.",
            "sadness": "The text reflects feelings of sorrow or loss.",
            "anger": "The text indicates frustration or hostility.",
            "fear": "The text shows apprehension or concern.",
            "surprise": "The text suggests an unexpected realization.",
            "depression": "The text shows signs of deep emotional vacuum or persistent low mood.",
            "anxiety": "The text indicates significant worry, dread, or apprehension.",
            "stress": "The text reflects overwhelming pressure or mental strain.",
            "bipolar": "The text suggests unpredictable energy shifts or emotional fluctuations.",
            "positive": "The text shows a constructive and optimistic outlook."
        }
        key_logic = logic_templates.get(top_label, "The analysis detected patterns associated with " + top_label + ".")

        return {
            "text": text,
            "processed_text": processed_text,
            "predictions": results,
            "sentiment_results": sentiment_results,
            "mental_health_results": mental_health_results,
            "top_prediction": results[0],
            "overall_sentiment": overall_sentiment,
            "sentiment_score": sentiment_score,
            "mental_health_indicator": mh_indicator,
            "mh_alert": mh_alert,
            "key_logic": key_logic
        }

# Singleton instance
service = None

def get_model_service(force_reload=False):
    global service
    if service is None or force_reload:
        service = ModelService()
    return service
