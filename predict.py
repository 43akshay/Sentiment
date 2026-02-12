import sys
import argparse
import json
from model_service import get_model_service

def main():
    parser = argparse.ArgumentParser(description="Sentiment Analysis CLI Prediction Tool")
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
        print("-" * 30)
        print(f"Top Prediction: {result['top_prediction']['label']}")
        print(f"Confidence: {result['top_prediction']['confidence']:.4f}")
        print("-" * 30)
        print("Emotion Probabilities:")
        for pred in result['predictions']:
            print(f"  {pred['label']:<10}: {pred['confidence']:.4f}")
        print("-" * 30)

if __name__ == "__main__":
    main()
