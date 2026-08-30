"""
Emotion recognition project - Stage 1
Data loading, cleaning, preprocessing, and traditional ML models
(Logistic Regression, Multinomial Naive Bayes, Linear SVM)

Dataset: combined DailyDialog + ISEAR + Emotion-Stimulus emotion-labelled text
Source: https://github.com/lukasgarbas/nlp-text-emotion
"""

import re
import json
import numpy as np
import pandas as pd
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                              classification_report, confusion_matrix)

RANDOM_STATE = 42
STOPWORDS = set(stopwords.words('english'))

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
train_df = pd.read_csv('data_train.csv')
test_df = pd.read_csv('data_test.csv')
train_df.columns = [c.strip() for c in train_df.columns]
test_df.columns = [c.strip() for c in test_df.columns]

print("Raw train shape:", train_df.shape)
print("Raw test shape:", test_df.shape)

# ---------------------------------------------------------------
# 2. Cleaning: drop duplicates / missing values
# ---------------------------------------------------------------
for df in (train_df, test_df):
    df.dropna(subset=['Text', 'Emotion'], inplace=True)
    df.drop_duplicates(subset=['Text'], inplace=True)
    df['Emotion'] = df['Emotion'].str.strip()
    df['Text'] = df['Text'].astype(str).str.strip()

print("After cleaning - train:", train_df.shape, "test:", test_df.shape)
print("\nLabel distribution (train):")
print(train_df['Emotion'].value_counts())

# ---------------------------------------------------------------
# 3. Text normalisation: lowercase, remove URLs/punctuation/digits,
#    tokenise, remove stopwords
# ---------------------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return ' '.join(tokens)

train_df['clean_text'] = train_df['Text'].apply(clean_text)
test_df['clean_text'] = test_df['Text'].apply(clean_text)

# drop rows that became empty after cleaning
train_df = train_df[train_df['clean_text'].str.len() > 0]
test_df = test_df[test_df['clean_text'].str.len() > 0]

X_train_text, y_train = train_df['clean_text'].values, train_df['Emotion'].values
X_test_text, y_test = test_df['clean_text'].values, test_df['Emotion'].values

print("\nFinal train size:", len(X_train_text), " Final test size:", len(X_test_text))

# ---------------------------------------------------------------
# 4. Feature extraction: TF-IDF
# ---------------------------------------------------------------
vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=2)
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)
print("TF-IDF matrix shape (train):", X_train.shape)

labels = sorted(pd.unique(np.concatenate([y_train, y_test])))
print("Classes:", labels)

# ---------------------------------------------------------------
# 5. Train + evaluate traditional ML models
# ---------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, C=5, random_state=RANDOM_STATE),
    "Multinomial Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC(random_state=RANDOM_STATE, C=1.0),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

    results[name] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm.tolist(),
        "labels": labels,
        "per_class_report": report,
    }
    print(f"\n=== {name} ===")
    print(f"Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))

# ---------------------------------------------------------------
# 6. Save everything needed for the report + deep learning stage
# ---------------------------------------------------------------
with open('ml_results.json', 'w') as f:
    json.dump(results, f, indent=2)

train_df[['Emotion', 'clean_text']].to_csv('train_clean.csv', index=False)
test_df[['Emotion', 'clean_text']].to_csv('test_clean.csv', index=False)

print("\nSaved: ml_results.json, train_clean.csv, test_clean.csv")
