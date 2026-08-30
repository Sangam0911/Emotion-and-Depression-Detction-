import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import json

train_df = pd.read_csv('train_clean.csv').dropna()
test_df = pd.read_csv('test_clean.csv').dropna()

vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=2)
X_train = vectorizer.fit_transform(train_df['clean_text'])
X_test = vectorizer.transform(test_df['clean_text'])

model = LogisticRegression(max_iter=1000, C=5, random_state=42)
model.fit(X_train, train_df['Emotion'])
preds = model.predict(X_test)

test_df = test_df.reset_index(drop=True)
test_df['pred'] = preds

# Get original (uncleaned) text for readability where possible
orig_test = pd.read_csv('data_test.csv')
orig_test.columns = [c.strip() for c in orig_test.columns]

wrong = test_df[test_df['Emotion'] != test_df['pred']]

# neutral -> joy confusions
neu_joy = wrong[(wrong['Emotion'] == 'neutral') & (wrong['pred'] == 'joy')]
anger_sad = wrong[(wrong['Emotion'] == 'anger') & (wrong['pred'] == 'sadness')]

print("=== neutral misclassified as joy (n=%d) ===" % len(neu_joy))
for t in neu_joy['clean_text'].head(6):
    print(" -", t)

print("\n=== anger misclassified as sadness (n=%d) ===" % len(anger_sad))
for t in anger_sad['clean_text'].head(6):
    print(" -", t)

print("\nTotal test:", len(test_df), "Total wrong:", len(wrong), "Error rate: %.2f%%" % (100*len(wrong)/len(test_df)))
