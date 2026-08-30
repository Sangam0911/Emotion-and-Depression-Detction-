# Emotion Recognition from Text — ML vs Deep Learning

An empirical comparison of traditional machine learning and deep learning
approaches to text-based emotion classification, built for the CN6000
final-year dissertation, *"Emotion Recognition from Text Using Machine
Learning and Deep Learning: An Empirical Comparison"* (Sangam Bhandari,
2662990, University of East London).

Rather than assume the common claim that deep learning always outperforms
traditional machine learning for text classification, this project tests
that assumption directly, under matched and controlled conditions, on a
dataset sized realistically for an individual student project — and reports
the result honestly, including the parts that didn't go as expected.

## Headline result

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| **Logistic Regression** | **68.11%** | 0.6828 | 0.6811 | 0.6818 |
| Linear SVM | 67.79% | 0.6796 | 0.6779 | 0.6786 |
| Multinomial Naive Bayes | 65.12% | 0.6596 | 0.6512 | 0.6384 |
| Bidirectional LSTM | 64.55% | 0.6489 | 0.6455 | 0.6461 |

The deep learning model did **not** outperform the simplest traditional
baseline on this dataset — it overfits sharply after epoch 2 (training
accuracy reaches 92% while validation accuracy plateaus around 65%). See
`results/figures/fig_lstm_training_curve.png` and Chapter 5–6 of the
dissertation for the full discussion of why.

## Dataset

11,330 English-language sentences labelled with one of five emotions
(anger, fear, joy, neutral, sadness), combining three peer-reviewed academic
corpora, compiled by Garbas (2020):

- **DailyDialog** (Li et al., 2017)
- **ISEAR** (Scherer & Wallbott, 1994)
- **Emotion-Stimulus** (Ghazi, Inkpen & Szpakowicz, 2015)

Source: [github.com/lukasgarbas/nlp-text-emotion](https://github.com/lukasgarbas/nlp-text-emotion)

Raw files are in `data/raw/`; cleaned files (after the preprocessing
pipeline in `src/01_preprocess_and_ml.py`) are saved to `data/processed/`.

## Repository structure

```
.
├── data/
│   ├── raw/                  # original dataset files
│   └── processed/            # cleaned dataset after preprocessing
├── src/
│   ├── 01_preprocess_and_ml.py   # cleaning, TF-IDF, 3 traditional ML models
│   ├── 02_deep_learning.py       # Bidirectional LSTM (TensorFlow/Keras)
│   ├── 03_make_figures.py        # generates all report figures
│   └── 04_error_analysis.py      # qualitative misclassification analysis
├── results/
│   ├── metrics/               # JSON/CSV results from every model
│   └── figures/               # generated charts (PNG)
├── notebooks/                 # (optional) place to explore interactively
├── requirements.txt
└── README.md
```

## How to reproduce

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

cd src
python 01_preprocess_and_ml.py     # cleans data, trains LR / NB / SVM
python 02_deep_learning.py         # trains the Bidirectional LSTM
python 03_make_figures.py          # generates all figures + summary table
python 04_error_analysis.py        # prints real misclassified examples
```

All random seeds are fixed (42), so results should reproduce exactly on the
same package versions (see `requirements.txt`).

## Ethical note

This dataset is public, peer-reviewed academic research data with no
directly identifying personal information. The broader class of application
this project contributes toward — automated inference about emotional state
from text — should be treated as a decision-support aid only, never a
diagnostic tool. See Chapter 6 of the dissertation for the full discussion.

## Citation

If referencing this project, please also cite the underlying datasets:

- Li, Y. et al. (2017) *DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset*. IJCNLP.
- Scherer, K.R. and Wallbott, H.G. (1994) *Evidence for universality and cultural variation of differential emotion response patterning*. Journal of Personality and Social Psychology, 66(2).
- Ghazi, D., Inkpen, D. and Szpakowicz, S. (2015) *Detecting Emotion Stimuli in Emotion-Bearing Sentences*. CICLing.
