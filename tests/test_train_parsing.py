import os
from preprocess import preprocess_text

def test_data_parsing():
    line = "i feel lucky;joy"
    parts = line.strip().rsplit(';', 1)
    assert len(parts) == 2
    text, label = parts
    assert text == "i feel lucky"
    assert label == "joy"

def test_data_parsing_with_semicolon():
    line = "i feel lucky; or do i?;joy"
    parts = line.strip().rsplit(';', 1)
    assert len(parts) == 2
    text, label = parts
    assert text == "i feel lucky; or do i?"
    assert label == "joy"
