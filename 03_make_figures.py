import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open('ml_results.json') as f:
    ml = json.load(f)
with open('dl_results.json') as f:
    dl = json.load(f)

all_results = {**ml, **dl}
model_names = list(all_results.keys())
labels = all_results[model_names[0]]['labels']

plt.rcParams.update({'font.size': 11})

# ---------------------------------------------------------------
# 1. Model comparison bar chart (Accuracy, Precision, Recall, F1)
# ---------------------------------------------------------------
metrics = ['accuracy', 'precision', 'recall', 'f1']
x = np.arange(len(model_names))
width = 0.2
fig, ax = plt.subplots(figsize=(9, 5))
for i, m in enumerate(metrics):
    vals = [all_results[n][m] for n in model_names]
    ax.bar(x + i * width, vals, width, label=m.capitalize())
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(model_names, rotation=10)
ax.set_ylim(0, 1)
ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison on Test Set')
ax.legend(loc='lower right')
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('fig_model_comparison.png', dpi=150)
plt.close()

# ---------------------------------------------------------------
# 2. Confusion matrices (one grid, all 4 models)
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 10))
for ax, name in zip(axes.flat, model_names):
    cm = np.array(all_results[name]['confusion_matrix'])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(name)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i,j]}", ha='center', va='center',
                     color='white' if cm_norm[i, j] > 0.5 else 'black', fontsize=8)
fig.suptitle('Confusion Matrices (test set)', fontsize=14)
plt.tight_layout()
plt.savefig('fig_confusion_matrices.png', dpi=150)
plt.close()

# ---------------------------------------------------------------
# 3. LSTM training curves (accuracy & loss)
# ---------------------------------------------------------------
hist = dl['Bidirectional LSTM']['history']
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].plot(hist['accuracy'], marker='o', label='Train')
axes[0].plot(hist['val_accuracy'], marker='o', label='Validation')
axes[0].set_title('LSTM Accuracy per Epoch')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(alpha=0.4)

axes[1].plot(hist['loss'], marker='o', label='Train')
axes[1].plot(hist['val_loss'], marker='o', label='Validation')
axes[1].set_title('LSTM Loss per Epoch')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(alpha=0.4)
plt.tight_layout()
plt.savefig('fig_lstm_training_curve.png', dpi=150)
plt.close()

# ---------------------------------------------------------------
# 4. Class distribution chart
# ---------------------------------------------------------------
import pandas as pd
train_df = pd.read_csv('data_train.csv')
train_df.columns = [c.strip() for c in train_df.columns]
counts = train_df['Emotion'].str.strip().value_counts()
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.bar(counts.index, counts.values, color='#4C72B0')
ax.set_title('Training Set Class Distribution')
ax.set_ylabel('Number of samples')
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('fig_class_distribution.png', dpi=150)
plt.close()

print("Saved 4 figures.")

# ---------------------------------------------------------------
# 5. Summary table (for easy insertion into the report)
# ---------------------------------------------------------------
summary_rows = []
for name in model_names:
    r = all_results[name]
    summary_rows.append({
        'Model': name,
        'Accuracy': round(r['accuracy'], 4),
        'Precision': round(r['precision'], 4),
        'Recall': round(r['recall'], 4),
        'F1-score': round(r['f1'], 4),
    })
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv('summary_results.csv', index=False)
print(summary_df)
