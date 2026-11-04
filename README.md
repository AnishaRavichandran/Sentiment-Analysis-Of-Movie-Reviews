# 🎬 Movie Review Sentiment Analysis using Bidirectional LSTM

A deep learning project that classifies movie reviews as **positive** or **negative** using a Bidirectional LSTM network, with hyperparameters tuned automatically via Bayesian Optimization.

---

## 📌 Business Goal

Movie studios, streaming platforms and review websites receive thousands of audience reviews every day. Reading them manually is slow and expensive.

This project automates that process so that businesses can:

- Quickly understand how audiences feel about a movie
- Track public opinion after a release or a trailer launch
- Spot negative feedback early and respond to it
- Support decisions on marketing, recommendations and content investment

---

## 🎯 Aim of the Project

To build an accurate and efficient **sentiment classification model** that reads a raw movie review and predicts whether the sentiment is positive or negative.

Key objectives:

1. Clean and prepare raw review text for modelling
2. Build a Bidirectional LSTM model that understands context in both directions of a sentence
3. Use Bayesian Optimization (Keras Tuner) to find the best hyperparameters
4. Evaluate the model on unseen test data using precision, recall and F1-score

---

## 📂 Dataset

- **Source:** Kaggle – Movie Reviews dataset
- **Size:** 25,000 rows
- **Columns:** review text and sentiment label
- **Labels:** positive / negative (the label mapping in the code also supports a neutral class)

**Data split (stratified, so class balance is preserved):**

| Split      | Share | Samples |
|------------|-------|---------|
| Training   | 70%   | 17,500  |
| Validation | 15%   | 3,750   |
| Testing    | 15%   | 3,750   |

---

## 🧹 Preprocessing Steps

1. **Load data** from the CSV file (text + sentiment)
2. **Lowercase** all text
3. **Remove punctuation** using regular expressions
4. **Remove stopwords** (common words such as "the", "is", "and") using NLTK
5. **Tokenization** – convert words to integer indices using the Keras `Tokenizer` (vocabulary limited to 5,000 words, with an out-of-vocabulary token)
6. **Padding** – pad all sequences to the same length (`padding='post'`)
7. **Label encoding** – positive → 0, negative → 1, neutral → 2
8. **Stratified train / validation / test split** (70 / 15 / 15)
9. **Class weights** computed from training labels only, to avoid data leakage and handle any class imbalance

---

## 🏗️ Model Architecture

```
Input (padded sequence of word indices)
        │
        ▼
Embedding layer
        │
        ▼
Bidirectional LSTM (return_sequences=True)
        │
        ▼
GlobalMaxPooling1D
        │
        ▼
Dense layer (ReLU)
        │
        ▼
Dropout
        │
        ▼
Dense output layer (Softmax)
```

**Why this design?**

- **Embedding** turns words into dense vectors that capture meaning
- **Bidirectional LSTM** reads the review forwards and backwards to understand context
- **GlobalMaxPooling** picks the strongest signal from the whole review
- **Dropout** reduces overfitting

**Hyperparameter tuning (Bayesian Optimization, 5 trials):**

| Hyperparameter     | Search range      |
|--------------------|-------------------|
| Embedding dimension| 64 – 256          |
| LSTM units         | 64 – 128          |
| Dense units        | 64 – 256          |
| Dropout rate       | 0.2 – 0.5         |
| Learning rate      | 1e-4 – 1e-2 (log) |

**Training setup:** Adam optimizer, Sparse Categorical Crossentropy loss, class weights, EarlyStopping and ReduceLROnPlateau callbacks.

---

## 📊 Results

Best validation accuracy during tuning: **~89.1%**

**Performance on the held-out test set (3,750 reviews):**

| Metric             | Score  |
|--------------------|--------|
| Accuracy           | 0.88   |
| Precision (weighted)| 0.883 |
| Recall (weighted)  | 0.881  |
| F1 Score (weighted)| 0.881  |
| F1 Score (macro)   | 0.881  |

**Classification report:**

| Class        | Precision | Recall | F1-score | Support |
|--------------|-----------|--------|----------|---------|
| Positive (0) | 0.86      | 0.92   | 0.89     | 1875    |
| Negative (1) | 0.91      | 0.85   | 0.88     | 1875    |

The model performs consistently on both classes, with no strong bias towards either sentiment.

---

## 🛠️ Tech Stack

- **Language:** Python
- **Deep Learning:** TensorFlow / Keras
- **Hyperparameter Tuning:** Keras Tuner (Bayesian Optimization)
- **NLP:** NLTK
- **Data and Metrics:** NumPy, scikit-learn
- **Environment:** Google Colab (GPU – T4)

---

## 🚀 Future Enhancements

- Use **transformer models** (BERT, DistilBERT) for higher accuracy
- Add **explainability** (LIME / SHAP) to show which words drove each prediction
- Extend to **multilingual** reviews

---

