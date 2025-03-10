# **Using RNN to Predict the next few sentences in a Paragraph**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gowtamyreddy/NLP/blob/main/RNN.ipynb)

## **Introduction**
Language modeling is an important task in **Natural Language Processing (NLP)**, and one powerful approach to this is using **Recurrent Neural Networks (RNNs)**. 

This project focuses on **predicting the next few sentences in a Paragraph** using an **RNN-based text generation model**. Given a sequence of words as input, the model learns from patterns in a dataset and tries to generate the most probable next word.

The model is trained on text data from **Harry Potter and the Sorcerer’s Stone**, allowing it to learn language structures and generate meaningful text. 



## **Steps in the Project**

### **1. Load the Data and Preprocess It**
- First, we take a text file (Harry Potter book in this case) and **read the content**.
- The text is then **converted to lowercase** to keep everything uniform.

---

### **2. Tokenization**
- The text is broken down into **words** and assigned unique numerical values.
- Any unknown words that were not seen in training are replaced with a special token `<OOV>` (Out Of Vocabulary).
- The text is then **converted into sequences of numbers** so the model can understand it.

---

### **3. Build the RNN Model**
- The model consists of three main layers:
  - **Embedding Layer**: Converts words into numerical vector representations.
  - **SimpleRNN Layers**: Helps the model understand the **sequence and flow** of words.
  - **Dense Layer with Softmax Activation**: Helps predict the **next possible word** in a sentence.

---

### **4. Prepare Input Sequences**
- The text is split into **smaller sequences** so the model can learn word patterns.
- Each sequence contains a **set of words**, and the model tries to predict the **next word** in the sequence.

---

### **5. Train the Model**
- The model is trained using **categorical cross-entropy**, a common loss function for classification tasks.
- It uses the **Adam optimizer**, which helps the model learn efficiently.
- The training process adjusts the model's internal weights so it gets better at predicting words.

---

### **6. Generate Text Using the Model**
- After training, the model can take a **starting sentence** and predict the **next words**.
- By continuing the predictions, it can generate **new sentences** in the style of the training text.

---

### **7. Example Output**
- After training, if you give the model a phrase like:
  
  **"Harry at Hogwarts"**
  
  It might generate something like:

  **"Harry at Hogwarts his hands had broken a sound of people whooshing and the movements of magic..."**

---


