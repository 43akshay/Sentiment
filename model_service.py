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

        # Calculate Overall Sentiment and Sentiment Score
        prob_dict = {res["label"]: res["confidence"] for res in results}

        pos_score = prob_dict.get("joy", 0) + prob_dict.get("love", 0)
        neg_score = prob_dict.get("anger", 0) + prob_dict.get("fear", 0) + prob_dict.get("sadness", 0)

        sentiment_score = pos_score - neg_score

        # Determine overall category
        if sentiment_score > 0.1:
            overall_sentiment = "Positive"
        elif sentiment_score < -0.1:
            overall_sentiment = "Negative"
        else:
            overall_sentiment = "Neutral"

        # Generate Key Logic (Simplified)
        top_label = results[0]["label"]
        logic_templates = {
            "joy": "The text expresses happiness, satisfaction, or pleasure, indicating a very positive emotional state.",
            "love": "The text conveys affection, warmth, or deep appreciation, showing strong positive connections.",
            "sadness": "The text reflects feelings of sorrow, disappointment, or loss, suggesting a negative emotional tone.",
            "anger": "The text indicates frustration, resentment, or hostility, pointing towards a strong negative sentiment.",
            "fear": "The text shows apprehension, anxiety, or concern, reflecting an unsettled or fearful state.",
            "surprise": "The text suggests amazement or an unexpected realization, which can be seen as a neutral or transitional emotion."
        }
        key_logic = logic_templates.get(top_label, "The analysis is based on the predominant emotional patterns detected in the text.")

        return {
            "text": text,
            "processed_text": processed_text,
            "predictions": results,
            "top_prediction": results[0],
            "overall_sentiment": overall_sentiment,
            "sentiment_score": sentiment_score,
            "key_logic": key_logic
        }

# Singleton instance
service = None

def get_model_service(force_reload=False):
    global service
    if service is None or force_reload:
        service = ModelService()
    return service
