from preprocess import preprocess_text

def test_lowercase():
    assert preprocess_text("HELLO") == "hello"

def test_trim():
    assert preprocess_text("  hello  ") == "hello"

def test_remove_urls():
    assert preprocess_text("check this http://google.com out") == "check this out"

def test_punctuation():
    # Our implementation replaces punctuation with space and collapses whitespace
    assert preprocess_text("hello, world!") == "hello world"

def test_complex():
    text = "  Check this AMAZING link: https://huggingface.co/ !!  "
    expected = "check this amazing link"
    assert preprocess_text(text) == expected
