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

    # Normalize punctuation
    # 1. Reduce multiple ! and ? to single ones
    text = re.sub(r'!+', '!', text)
    text = re.sub(r'\?+', '?', text)

    # 2. Keep apostrophes for contractions, but remove other punctuation except ! and ?
    # We replace most punctuation with spaces, but carefully.
    # We'll use a translation table or regex for efficiency.
    to_remove = string.punctuation.replace("'", "").replace("!", "").replace("?", "")
    pattern = f'[{re.escape(to_remove)}]'
    text = re.sub(pattern, ' ', text)

    # Trim whitespace and remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text
