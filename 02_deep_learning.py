"""
Emotion recognition project - Stage 2
Deep learning model: Bidirectional LSTM with a trainable embedding layer
"""

import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                              classification_report, confusion_matrix)

RANDOM_STATE = 42
tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# ---------------------------------------------------------------
# 1. Load cleaned data from stage 1
# ---------------------------------------------------------------
train_df = pd.read_csv('train_clean.csv')
test_df = pd.read_csv('test_clean.csv')
train_df.dropna(inplace=True)
test_df.dropna(inplace=True)

labels = sorted(train_df['Emotion'].unique())
le = LabelEncoder()
le.fit(labels)
y_train = le.transform(train_df['Emotion'])
y_test = le.transform(test_df['Emotion'])

# ---------------------------------------------------------------
# 2. Tokenise + pad sequences
# ---------------------------------------------------------------
MAX_VOCAB = 10000
MAX_LEN = 40

tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token='<OOV>')
tokenizer.fit_on_texts(train_df['clean_text'])

X_train_seq = tokenizer.texts_to_sequences(train_df['clean_text'])
X_test_seq = tokenizer.texts_to_sequences(test_df['clean_text'])

X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding='post', truncating='post')

vocab_size = min(MAX_VOCAB, len(tokenizer.word_index) + 1)
num_classes = len(labels)

# ---------------------------------------------------------------
# 3. Build Bidirectional LSTM model
# ---------------------------------------------------------------
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=100, input_length=MAX_LEN),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax'),
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    X_train_pad, y_train,
    validation_split=0.1,
    epochs=15,
    batch_size=32,
    callbacks=[early_stop],
    verbose=2,
)

# ---------------------------------------------------------------
# 4. Evaluate
# ---------------------------------------------------------------
y_pred_probs = model.predict(X_test_pad)
y_pred = np.argmax(y_pred_probs, axis=1)

acc = accuracy_score(y_test, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=labels, zero_division=0, output_dict=True)

print(f"\n=== Bidirectional LSTM ===")
print(f"Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")
print(classification_report(y_test, y_pred, target_names=labels, zero_division=0))

dl_results = {
    "Bidirectional LSTM": {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "labels": labels,
        "per_class_report": report,
        "history": {k: [float(x) for x in v] for k, v in history.history.items()},
        "vocab_size": int(vocab_size),
        "max_len": MAX_LEN,
        "trainable_params": int(model.count_params()),
    }
}

with open('dl_results.json', 'w') as f:
    json.dump(dl_results, f, indent=2)

print("\nSaved: dl_results.json")
