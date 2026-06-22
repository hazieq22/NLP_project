# Movie Review Sentiment Detector

This project is a Streamlit NLP application that predicts whether a movie review is positive or negative. It uses the IMDB Dataset of 50,000 labeled movie reviews and compares multiple machine learning pipelines.

## Project Objectives

- Build a complete NLP text classification pipeline.
- Preprocess movie review text using cleaning, tokenization, stopword removal, and stemming.
- Compare multiple feature extraction methods and machine learning models.
- Deploy a working Streamlit app for real-time sentiment prediction.
- Present visual insights about the dataset and model performance.

## Dataset

- Source: IMDB Dataset
- Size: 50,000 labeled reviews
- Labels: `pos` and `neg`
- Raw CSV format: `data/IMDB Dataset.csv`
- Project columns after loading: `text`, `label`, and `label_name`

The dataset can be checked by:

```bash
python scripts/prepare_data.py
```

## NLP Pipeline

Preprocessing steps:

1. Lowercase text
2. Remove URLs, HTML tags, special characters, and numbers
3. Tokenize text into words
4. Remove English stopwords while keeping negation words such as `not`
5. Stem words with Porter stemming

Feature extraction methods:

- Bag of Words
- TF-IDF
- TF-IDF word n-grams

Models trained:

- Naive Bayes
- Logistic Regression
- Tuned Linear SVM

Current best model:

- TF-IDF word n-grams + Linear SVM
- Accuracy: 91.92%
- Weighted F1-score: 91.92%

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

## Streamlit App Pages

The app includes all required sections:

- Home/About
- Text Analyzer
- Data Explorer
- Visualizations
- Model Info

## Visualizations

The app includes more than five visualizations:

- Positive review word cloud
- Negative review word cloud
- Sentiment/class distribution
- Confusion matrix
- Model comparison chart
- Top words by sentiment
- Top bigrams
- Review length distribution

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Prepare the dataset and train the models:

```bash
python scripts/prepare_data.py
python train_model.py
```

Run the app:

```bash
streamlit run app.py
```

Optional BERT comparison output:

```bash
.\download_bert_model.bat
```

Run this once to download the pretrained DistilBERT sentiment model, then restart Streamlit.

## Project Structure

```text
.
|-- app.py
|-- train_model.py
|-- requirements.txt
|-- README.md
|-- data/
|   `-- IMDB Dataset.csv
|-- models/
|   |-- model_bundle.joblib
|   `-- movie_sentiment_model.joblib
|-- notebooks/
|   `-- model_development.ipynb
|-- outputs/
|   |-- confusion_matrix.csv
|   `-- model_comparison.csv
|-- scripts/
|   `-- prepare_data.py
`-- src/
    |-- data_utils.py
    `-- nlp_utils.py
```

## Team Members

- Hazieq Aiman Bin Hamizi - A24AI0033
- Nik Muhammad Faris Bin Nik Zaki - A24AI0070
- Muhammad Hassan Naeik Bin Norizan - A24AI0056
- Nik Muhammad Muqri Hazim Bin Nik Nor Ahmarizam - A24AI0071

## References

- IMDB Dataset of 50K Movie Reviews.
- Scikit-learn documentation.
- Streamlit documentation.
