# Sentiment Analysis with Bi-LSTM + Keras Tuner

A deep learning project that classifies text into **positive**, **negative**, or **neutral** sentiment using a Bidirectional LSTM neural network, with hyperparameters automatically tuned via Keras Tuner.

## Overview

This project builds a text sentiment classifier trained on labeled text data. It uses:
- **Text preprocessing**: lowercasing, punctuation removal, and stopword removal (NLTK)
- **Tokenization & padding** with Keras `Tokenizer` (vocab size: 5000)
- **Bidirectional LSTM** architecture with an embedding layer, global max pooling, and a dense classification head
- **Keras Tuner (Random Search)** to automatically find the best embedding dimension, LSTM units, dense units, dropout rate, and learning rate
- **Class weighting** to handle imbalanced sentiment classes
- An interactive command-line loop to test the trained model on custom input text

## Model Architecture

```
Embedding → Bidirectional LSTM → GlobalMaxPooling1D → Dense (ReLU) → Dropout → Dense (Softmax, 3 classes)
```

Hyperparameters tuned:
| Hyperparameter    | Search Range        |
|--------------------|---------------------|
| Embedding dimension | 64 – 256 (step 32) |
| LSTM units          | 64 – 128 (step 32) |
| Dense units          | 64 – 256 (step 32) |
| Dropout rate         | 0.2 – 0.5 (step 0.1) |
| Learning rate         | 1e-4 – 1 (log scale) |

## Dataset

The notebook expects a zipped CSV dataset (`test.csv` inside a `.zip` file) with two columns:
1. Text
2. Sentiment label (`positive`/`pos`, `negative`/`neg`, `neutral`/`neu`)

> Update the `zip_file_path` variable in the notebook to point to your own dataset.



## Usage

1. Open `Sentimentanalysisfinal.ipynb` in Jupyter Notebook / Google Colab.
2. Update the dataset path to your zipped CSV file.
3. Run all cells to:
   - Preprocess the data
   - Search for the best hyperparameters
   - Train the best model
4. Once training completes, use the interactive prompt to test the model:
   ```
   Enter your message (or 'exit' to quit): This movie was amazing!
   Predicted sentiment: positive
   ```



## License

This project is open source and available under the [MIT License](LICENSE).
