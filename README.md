# Text Generation using RNN, LSTM, and Transformer

## Introduction

This project explores different deep learning techniques for **text generation** using **Recurrent Neural Networks (RNNs), Long Short-Term Memory (LSTM), and Transformer models**. The goal is to understand how each model processes text sequences, learns patterns, and generates coherent text.

Initially, an **RNN-based model** was implemented to generate text by learning sequential dependencies. However, due to limitations in capturing long-range dependencies, an **LSTM model** was introduced to improve text coherence and retain contextual understanding. Finally, a **Transformer model** was implemented, leveraging **self-attention mechanisms** to generate high-quality text predictions.

The dataset used for training is **Harry Potter and the Sorcerer’s Stone**, allowing the models to learn meaningful linguistic patterns and generate creative text based on input prompts.

## Approach

### **1. RNN-Based Text Generation**
- Used a **basic Recurrent Neural Network (RNN)** to predict the next words in a sequence.
- Encountered challenges in **long-term dependencies** and **vanishing gradients**.
- Performance was limited due to the inability of RNNs to store information over long sequences.

### **2. LSTM-Based Text Generation**
- Improved upon RNNs by using **LSTM (Long Short-Term Memory) networks**.
- LSTMs can **store and recall** information over longer sequences, reducing vanishing gradient issues.
- Enhanced the ability to generate **more contextually relevant** text.

### **3. Transformer-Based Text Generation**
- Implemented a **decoder-only Transformer model** for text generation.
- Transformers use **self-attention** to efficiently capture long-term dependencies.
- Achieved **higher quality and contextually aware** text predictions.

## Features

- **Comparison of RNN, LSTM, and Transformer** for text generation.
- **Uses word tokenization and sequence padding** for model input processing.
- **Trained on a real-world dataset** (Harry Potter book) for natural text prediction.
- **Generates text based on input prompts** with different architectures.

## Dependencies

Ensure you have the following libraries installed before running the notebooks:

```bash
pip install tensorflow numpy keras
```

## Dataset

The dataset used for training is available on **Kaggle**: [Harry Potter Books Dataset](https://www.kaggle.com/datasets/shubhammaindola/harry-potter-books)

### Preprocessing Steps

1. **Load the text data** and convert it to lowercase.
2. **Tokenize the text**, assigning unique numerical values to each word.
3. **Convert text into sequences** for input into the models.
4. **Apply padding** to standardize sequence lengths.
5. **One-hot encode** target words for categorical prediction.

## Model Architectures

### **1. RNN Model**
- **Embedding Layer**: Converts words into dense vector representations.
- **Simple RNN Layer**: Captures sequential dependencies.
- **Dense Layer with Softmax Activation**: Predicts the next word in the sequence.

### **2. LSTM Model**
- **Embedding Layer**: Converts words into vector representations.
- **Two LSTM Layers**: Retains long-term dependencies.
- **Dense Layer with Softmax Activation**: Predicts the next word.

### **3. Transformer Model**
- **Token & Positional Embedding Layer**: Converts words and positions into embeddings.
- **Multi-Head Self-Attention Layer**: Captures dependencies between words.
- **Feed-Forward Network (FFN)**: Adds non-linearity and learning capacity.
- **Dense Layer with Softmax Activation**: Predicts the next word.

## Sample Text Generation

After training, the models generate text based on an input seed phrase. Example:

```python
seed_text = "Harry at Hogwarts"
generated_text = generate_text(seed_text, next_words=50, temperature=0.7)
print(generated_text)
```

### Example Output
```
"Harry at Hogwarts narrowly fantastic touching… weasleys cheers lunchtime possible below doom reflection..."
```

## Running the Notebooks

You can open and run each notebook in Google Colab using the following links:

- **RNN-Based Model**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gowtamyreddy/NLP/blob/main/RNN.ipynb)
- **LSTM-Based Model**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gowtamyreddy/NLP/blob/main/LSTM.ipynb)
- **Transformer-Based Model**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gowtamyreddy/NLP/blob/main/transformer.ipynb)

## Future Enhancements

- Experiment with **Bidirectional LSTMs** for improved sequence learning.
- Fine-tune models using **larger text datasets**.
- Implement **attention mechanisms in LSTMs** for better contextual understanding.
- Extend the project by exploring **GPT-based architectures**.


