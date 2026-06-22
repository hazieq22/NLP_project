from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import nltk


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NLTK_DATA_DIR = PROJECT_ROOT / "nltk_data"
NLTK_DATA_DIR.mkdir(exist_ok=True)

if str(NLTK_DATA_DIR) not in nltk.data.path:
    nltk.data.path.insert(0, str(NLTK_DATA_DIR))


def ensure_nltk_resources() -> None:
    """Download optional NLTK resources when internet access is available."""
    resources = {
        "corpora/stopwords": "stopwords",
    }

    for resource_path, package_name in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(package_name, download_dir=str(NLTK_DATA_DIR), quiet=True, raise_on_error=False)


@lru_cache(maxsize=1)
def _stopwords() -> set[str]:
    ensure_nltk_resources()
    keep_words = {"not", "no", "nor", "never"}
    fallback = {
        "a",
        "about",
        "after",
        "all",
        "also",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "but",
        "by",
        "can",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "his",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "just",
        "me",
        "more",
        "most",
        "my",
        "of",
        "on",
        "or",
        "our",
        "she",
        "so",
        "than",
        "that",
        "the",
        "their",
        "them",
        "there",
        "they",
        "this",
        "to",
        "too",
        "was",
        "we",
        "were",
        "what",
        "when",
        "which",
        "who",
        "will",
        "with",
        "you",
        "your",
    }
    try:
        from nltk.corpus import stopwords

        return set(stopwords.words("english")) - keep_words
    except LookupError:
        return fallback - keep_words


@lru_cache(maxsize=1)
def _stemmer():
    from nltk.stem import PorterStemmer

    return PorterStemmer()


def preprocess_text(text: str) -> str:
    """Clean, tokenize, remove stopwords, and stem review text."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    stops = _stopwords()
    stemmer = _stemmer()
    tokens = []
    for token in text.split():
        token = token.strip("'")
        if len(token) <= 1 or token in stops:
            continue
        tokens.append(stemmer.stem(token))
    return " ".join(tokens)


def clean_text_for_ngrams(text: str) -> str:
    """Light cleaning for high-performing TF-IDF n-gram models."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def count_words(texts: list[str], top_n: int = 25) -> list[dict[str, int | str]]:
    from collections import Counter

    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(preprocess_text(text).split())
    return [{"word": word, "count": count} for word, count in counter.most_common(top_n)]
