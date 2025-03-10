# LSTM-Based Text Generation Model



## Introduction

This project implements an **LSTM-based text generation model** using **TensorFlow and Keras**. The model is trained on text data from **Harry Potter and the Sorcerer’s Stone**, aiming to predict the next words in a sequence based on learned patterns.

Recurrent Neural Networks (RNNs), particularly **Long Short-Term Memory (LSTM) networks**, are well-suited for sequence prediction tasks. This project utilizes an **LSTM-based deep learning model** to generate text sequences after training on a given dataset.

## Features

- Implements **LSTM neural networks** for text generation
- Uses **word tokenization and sequence padding** for input processing
- Trained on the **Harry Potter book dataset**
- Generates **contextually relevant text** given an input prompt

## Dependencies

Ensure you have the following libraries installed before running the notebook:

```bash
pip install tensorflow numpy keras
```

## Dataset

The dataset used for training is available on **Kaggle**: [Harry Potter Books Dataset](https://www.kaggle.com/datasets/shubhammaindola/harry-potter-books)

### Preprocessing Steps

1. **Load the text data** and convert it to lowercase.
2. **Tokenize the text**, assigning unique numerical values to each word.
3. **Convert text into sequences** for input into the model.
4. **Apply padding** to standardize sequence lengths.
5. **One-hot encode** target words for categorical prediction.

## Model Architecture

The LSTM model is designed using TensorFlow’s **Keras Sequential API** and consists of:

- **Embedding Layer**: Converts words into dense vector representations.
- **Two LSTM Layers**: Captures sequential dependencies in text.
- **Dense Layer with Softmax Activation**: Predicts the next word in the sequence.

## Training

The model is compiled using:

- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam
- **Epochs**: 20
- **Batch Size**: 128

Training is performed to adjust the model’s internal weights and improve prediction accuracy.

## Sample Text Generation

After training, the model generates text based on an input seed phrase. Example:

```python
seed_text = "Harry at Hogwarts"
generated_text = generate_text(seed_text, next_words=50, temperature=0.7)
print(generated_text)
```

### Example Output

```
"Harry at Hogwarts narrowly fantastic touching… weasleys cheers lunchtime possible below doom reflection..."
```

## Running the Notebook

You can open and run the notebook in Google Colab using the following link:



## Future Enhancements

- Experiment with **Bidirectional LSTMs** for improved sequence learning.
- Fine-tune the model using **larger text datasets**.
- Implement **attention mechanisms** for better contextual understanding.




