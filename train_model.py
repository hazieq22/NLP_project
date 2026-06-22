from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.data_utils import DATA_PATH, load_or_create_dataset
from src.nlp_utils import clean_text_for_ngrams, count_words, preprocess_text


RANDOM_STATE = 42
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")


def make_vectorizers() -> dict[str, object]:
    required_preprocessing = {
        "preprocessor": preprocess_text,
        "tokenizer": str.split,
        "token_pattern": None,
        "lowercase": False,
        "min_df": 2,
        "max_df": 0.95,
    }
    return {
        "Bag of Words": CountVectorizer(**required_preprocessing, max_features=12000),
        "TF-IDF": TfidfVectorizer(**required_preprocessing, max_features=12000),
        "TF-IDF N-grams": TfidfVectorizer(
            preprocessor=clean_text_for_ngrams,
            token_pattern=r"(?u)\b[a-z][a-z']+\b",
            min_df=2,
            max_df=0.95,
            max_features=700000,
            ngram_range=(1, 4),
            sublinear_tf=True,
        ),
    }


def make_models() -> dict[str, object]:
    return {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1200, random_state=RANDOM_STATE),
        "Linear SVM": LinearSVC(C=1.5, random_state=RANDOM_STATE),
    }


def build_pipelines() -> dict[str, Pipeline]:
    vectorizers = make_vectorizers()
    models = make_models()
    return {
        "BoW + Naive Bayes": Pipeline(
            [("vectorizer", vectorizers["Bag of Words"]), ("classifier", models["Naive Bayes"])]
        ),
        "TF-IDF + Naive Bayes": Pipeline(
            [("vectorizer", vectorizers["TF-IDF"]), ("classifier", MultinomialNB())]
        ),
        "TF-IDF + Logistic Regression": Pipeline(
            [("vectorizer", vectorizers["TF-IDF"]), ("classifier", models["Logistic Regression"])]
        ),
        "TF-IDF N-grams + Linear SVM": Pipeline(
            [("vectorizer", vectorizers["TF-IDF N-grams"]), ("classifier", models["Linear SVM"])]
        ),
    }


def summarize_dataset(df: pd.DataFrame) -> dict[str, object]:
    text_lengths = df["text"].str.split().str.len()
    return {
        "total_reviews": int(len(df)),
        "labels": df["label"].value_counts().to_dict(),
        "average_words": round(float(text_lengths.mean()), 2),
        "median_words": round(float(text_lengths.median()), 2),
        "min_words": int(text_lengths.min()),
        "max_words": int(text_lengths.max()),
    }


def make_top_words(df: pd.DataFrame) -> dict[str, list[dict[str, int | str]]]:
    return {
        "Positive": count_words(df.loc[df["label"] == "pos", "text"].tolist(), top_n=30),
        "Negative": count_words(df.loc[df["label"] == "neg", "text"].tolist(), top_n=30),
    }


def make_top_ngrams(df: pd.DataFrame, top_n: int = 20) -> list[dict[str, int | str]]:
    counter: Counter[str] = Counter()
    for text in df["text"]:
        tokens = preprocess_text(text).split()
        counter.update(" ".join(tokens[i : i + 2]) for i in range(len(tokens) - 1))
    return [{"ngram": phrase, "count": count} for phrase, count in counter.most_common(top_n)]


def train() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = load_or_create_dataset(DATA_PATH)
    df = df.dropna(subset=["text", "label"]).copy()
    df = df[df["label"].isin(["pos", "neg"])]

    x_train, x_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )

    results = []
    reports = {}
    trained_models = {}
    for name, pipeline in build_pipelines().items():
        print(f"Training {name}...")
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        )
        accuracy = accuracy_score(y_test, predictions)
        results.append(
            {
                "model": name,
                "accuracy": round(float(accuracy), 4),
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "f1_score": round(float(f1), 4),
            }
        )
        reports[name] = classification_report(y_test, predictions, output_dict=True, zero_division=0)
        trained_models[name] = pipeline

    metrics_df = pd.DataFrame(results).sort_values(
        by=["f1_score", "accuracy"],
        ascending=False,
    )
    best_model_name = str(metrics_df.iloc[0]["model"])
    best_model = trained_models[best_model_name]
    best_predictions = best_model.predict(x_test)
    labels = ["neg", "pos"]
    cm = confusion_matrix(y_test, best_predictions, labels=labels)

    metrics_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(OUTPUT_DIR / "confusion_matrix.csv")

    bundle = {
        "model": best_model,
        "best_model_name": best_model_name,
        "metrics": metrics_df.to_dict(orient="records"),
        "classification_reports": reports,
        "confusion_matrix": cm.tolist(),
        "labels": labels,
        "label_names": {"neg": "Negative", "pos": "Positive"},
        "dataset_stats": summarize_dataset(df),
        "top_words": make_top_words(df),
        "top_ngrams": make_top_ngrams(df),
        "test_size": len(x_test),
        "train_size": len(x_train),
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "preprocessing_steps": [
            "Lowercase text",
            "Remove URLs, HTML tags, special characters, and numbers",
            "Tokenize into words",
            "Remove English stopwords while keeping negation words for baseline models",
            "Stem words with Porter stemming for baseline models",
            "Use light cleaning for the tuned SVM so sentiment phrases such as 'not good' remain available to n-grams",
        ],
        "feature_methods": ["Bag of Words", "TF-IDF", "TF-IDF word n-grams"],
        "models_trained": list(build_pipelines().keys()),
    }
    joblib.dump(bundle, MODEL_DIR / "model_bundle.joblib")
    joblib.dump(best_model, MODEL_DIR / "movie_sentiment_model.joblib")

    print("\nTraining complete.")
    print(f"Best model: {best_model_name}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    train()
