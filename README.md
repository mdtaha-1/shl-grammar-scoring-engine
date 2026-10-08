# SHL Grammar Scoring Engine

A multimodal machine learning pipeline for predicting grammar scores from spoken English audio.

This project was developed for the **SHL Hiring Assessment 2026** competition, where the task is to predict a grammar score from 0–5 for each spoken-audio recording.

## Approach

The final system combines three complementary sources of information:

- **Wav2Vec2** — pretrained speech representations using mean and standard-deviation pooling.
- **Acoustic features** — duration, energy, spectral characteristics, zero-crossing behaviour, speech activity, silence, and pause statistics.
- **Whisper-small transcript features** — selected lexical and sentence-level features extracted from automatic speech recognition transcripts.

These features are combined and passed to a **CatBoost Regressor**.

## Final Feature Set

The model uses:

- 1,536 Wav2Vec2 speech features
- 18 handcrafted acoustic features
- 4 transcript features:
  - `word_count`
  - `unique_word_ratio`
  - `chars_per_word`
  - `sentence_length_std`

**Total: 1,558 features**

## Model

The final CatBoost configuration:

- Iterations: 700
- Depth: 6
- Learning rate: 0.03
- L2 regularization: 5
- Loss: RMSE
- Random seed: 42

Model selection was evaluated using **5-fold cross-validation** with RMSE and Pearson correlation.

## Results

| Metric | Score |
|---|---:|
| 5-Fold CV RMSE | **0.6608** |
| 5-Fold CV Pearson | **0.8466** |
| Final training RMSE | **0.2976** |
| Kaggle leaderboard score | **0.5073** |

The final model was trained on the complete training set and used to generate predictions for the competition test set.

## Repository Contents

```text
shl-grammar-scoring-engine/
├── README.md
└── shl_grammar_scoring_clean.ipynb
