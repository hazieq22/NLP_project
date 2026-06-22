import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import DATA_PATH, find_dataset, load_or_create_dataset


if __name__ == "__main__":
    source = find_dataset(DATA_PATH)
    df = load_or_create_dataset(source)
    print(f"Dataset found: {source}")
    print(f"Rows after cleaning: {len(df):,}")
    print("Columns used by project: text, label, label_name")
    print(df["label_name"].value_counts().to_string())
