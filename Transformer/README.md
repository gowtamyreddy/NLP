# **Transformer-Based Text Generation Model**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gowtamyreddy/NLP/blob/main/transformer.ipynb)

## **Introduction**
This project implements a **Transformer-based text generation model** using **TensorFlow and Keras**. The model is designed in a **decoder-only format**, excluding the masking mechanism from a regular decoder. This allows the model to access future words during training. The project utilizes a **single Transformer block** to process and generate text.

The dataset used for training is the **Harry Potter book series**, specifically **"Harry Potter and the Sorcerer’s Stone"**. The text is tokenized and trained to predict the next words based on input sequences.

## **Features**
- Uses a **Transformer decoder model** (without masking)
- Implements **Multi-Head Self-Attention and Feed-Forward Neural Networks**
- Utilizes **Token and Positional Embedding layers**
- Trained on **Harry Potter books dataset** for text prediction
- Generates **contextually relevant text** based on input prompts

## **Dependencies**
Ensure you have the following libraries installed before running the notebook:

```bash
pip install tensorflow numpy keras
```

## **Dataset**
The dataset used for training is available on **Kaggle**: [Harry Potter Books Dataset](https://www.kaggle.com/datasets/shubhammaindola/harry-potter-books)

### **Preprocessing Steps**
1. Load the text data and convert it to lowercase.
2. Tokenize the text and create sequences of a fixed length (50 words).
3. Apply **padding** to ensure consistent input shape.
4. Convert target words into **one-hot encoding** for categorical prediction.

## **Model Architecture**
The model is designed using TensorFlow’s **Keras Functional API**. It consists of:

- **Token & Positional Embedding Layer**: Converts words and positions into embeddings.
- **Multi-Head Self-Attention Layer**: Captures dependencies between words.
- **Feed-Forward Network (FFN)**: Adds non-linearity and learning capacity.
- **Output Dense Layer**: Uses **Softmax activation** to predict the next word.

## **Training**
The model is compiled with **Adam optimizer** and **Categorical Crossentropy loss**. Training is performed for **15 epochs** with a batch size of **32**.

## **Sample Text Generation**
After training, the model can generate text based on an input seed phrase. Example:

```python
seed_text = "Harry at Hogwarts"
generated_text = generate_text(seed_text, next_words=50, max_sequence_len=51)
print(generated_text)
```

### **Example Output**
```
"Harry at Hogwarts and he heard of course books the first few days ..."
```

## **Running the Notebook**
You can open and run the notebook in Google Colab using the following link:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gowtamyreddy/NLP/blob/main/transformer.ipynb)

## **Future Enhancements**
- Implement **Masked Self-Attention** for standard Transformer decoding.
- Train on larger datasets for improved text coherence.
- Fine-tune the model using **GPT-based architectures**.


