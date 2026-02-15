import sys
import argparse
import json
from model_service import get_model_service

def main():
    parser = argparse.ArgumentParser(description="SentiMind: Mental Health & Sentiment Prediction Tool")
    parser.add_argument("text", type=str, help="Text to analyze")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    args = parser.parse_args()

    service = get_model_service()
    result = service.predict(args.text)

    if "error" in result:
        print(f"Error: {result['error']}")
        sys.exit(1)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\nInput Text: {result['text']}")
        print("-" * 50)
        print(f"Overall Sentiment: {result['overall_sentiment']} (Score: {result['sentiment_score']:.2f})")
        print(f"Mental Health Indicator: {result['mental_health_indicator']}")
        print("-" * 50)
        print(f"Top Prediction: {result['top_prediction']['label']} ({result['top_prediction']['confidence']:.4f})")
        print("-" * 50)
        print("Detailed Label Probabilities:")
        for pred in result['predictions']:
            print(f"  {pred['label']:<15}: {pred['confidence']:.4f}")
        print("-" * 50)

if __name__ == "__main__":
    main()
