import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import json
import os
import argparse
from preprocess import preprocess_text
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train(data_path, epochs=3, batch_size=16):
    # Load data
    texts = []
    labels = []
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found.")
        return

    with open(data_path, 'r') as f:
        for line in f:
            if ';' in line:
                parts = line.strip().rsplit(';', 1)
                if len(parts) == 2:
                    text, label = parts
                    texts.append(preprocess_text(text))
                    labels.append(label)

    if not texts:
        print("Error: No data found in the file.")
        return

    unique_labels = sorted(list(set(labels)))
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {str(i): label for label, i in label2id.items()}

    num_labels = len(unique_labels)
    label_ids = [label2id[l] for l in labels]

    # Create dataset
    df = pd.DataFrame({'text': texts, 'label': label_ids})
    dataset = Dataset.from_pandas(df)

    # Ensure we have enough samples to split
    if len(dataset) > 1:
        dataset = dataset.train_test_split(test_size=min(0.2, 1.0/len(dataset)))
        train_ds = dataset["train"]
        test_ds = dataset["test"]
    else:
        train_ds = dataset
        test_ds = dataset

    model_name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

    train_tokenized = train_ds.map(tokenize_function, batched=True)
    test_tokenized = test_ds.map(tokenize_function, batched=True)

    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

    output_dir = "./models/sentiment_model"
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir="./outputs",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=0,
        weight_decay=0.01,
        logging_dir="./outputs/logs",
        logging_steps=1,
        evaluation_strategy="epoch" if len(dataset) > 1 else "no",
        save_strategy="epoch" if len(dataset) > 1 else "no",
        load_best_model_at_end=True if len(dataset) > 1 else False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=test_tokenized,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Save model and tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save label mapping explicitly just in case
    with open(os.path.join(output_dir, "label_mapping.json"), "w") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f)

    # Save final metrics
    eval_results = trainer.evaluate()
    with open("./outputs/metrics.json", "w") as f:
        json.dump(eval_results, f)

    print(f"Training complete. Model saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/train.txt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    train(args.data, args.epochs, args.batch_size)
