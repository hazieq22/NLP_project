from transformers import pipeline


MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"


def main() -> None:
    print("Downloading pretrained DistilBERT sentiment model...")
    classifier = pipeline(
        "text-classification",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
    )
    print("Download complete.")
    print(classifier("This movie was amazing.")[0])


if __name__ == "__main__":
    main()
