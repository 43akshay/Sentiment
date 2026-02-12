import re
import string

def preprocess_text(text: str) -> str:
    """
    Applies preprocessing steps to the input text:
    - Lowercase
    - Trim whitespace
    - Remove URLs
    - Normalize punctuation
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Normalize punctuation (remove or replace with standard)
    # Here we'll just keep it simple and ensure no weird characters are causing issues
    # but keeping basic punctuation is often good for transformers.
    # The PRD says "normalize punctuation", I'll remove excessive punctuation.
    text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)

    # Trim whitespace and remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text
