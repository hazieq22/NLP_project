from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data/IMDB Dataset.csv"


def find_dataset(path: Path = DATA_PATH) -> Path:
    """Find the IMDB dataset even if it was placed in the project root."""
    candidates = [
        path,
        PROJECT_ROOT / "IMDB Dataset.csv",
        PROJECT_ROOT / "data/imdb_dataset.csv",
        PROJECT_ROOT / "data/imdb_reviews.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find the IMDB dataset. Put 'IMDB Dataset.csv' inside the data folder."
    )


def normalize_imdb_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Convert common IMDB CSV column names into the required text/label format."""
    df = df.copy()
    if {"review", "sentiment"}.issubset(df.columns):
        df = df.rename(columns={"review": "text", "sentiment": "label"})
    elif not {"text", "label"}.issubset(df.columns):
        raise ValueError("Dataset must contain either review/sentiment or text/label columns.")

    df = df[["text", "label"]].dropna()
    df["text"] = df["text"].astype(str)
    df["label"] = (
        df["label"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"positive": "pos", "negative": "neg"})
    )
    df = df[df["label"].isin(["pos", "neg"])].copy()
    df["label_name"] = df["label"].map({"pos": "Positive", "neg": "Negative"})
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def load_or_create_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    dataset_path = find_dataset(path)
    return normalize_imdb_dataset(pd.read_csv(dataset_path))
