# Advanced Movie Sentiment Analysis Using NLP, Machine Learning and Deep Learning



## 📌 Project Overview



This project presents an advanced sentiment analysis system for classifying movie reviews as **Positive** or **Negative**.



The project extends a traditional TF-IDF and machine learning approach by comparing multiple machine learning and deep learning architectures, including:



- Logistic Regression

- Multinomial Naive Bayes

- Linear Support Vector Machine (SVM)

- Long Short-Term Memory (LSTM)

- Bidirectional LSTM (BiLSTM)

- DistilBERT Transformer



The final system uses a fine-tuned **DistilBERT transformer model** and provides a Streamlit-based interface for real-time sentiment prediction.



---



## 🎯 Objectives



The main objectives of this project are:



1\. Perform sentiment analysis on a large movie-review dataset.

2\. Apply classical machine learning techniques for sentiment classification.

3\. Implement recurrent neural networks using LSTM and BiLSTM.

4\. Fine-tune a transformer-based DistilBERT model.

5\. Compare the performance of different approaches.

6\. Perform error analysis using false positives and false negatives.

7\. Explore Explainable AI using Integrated Gradients and Captum.

8\. Develop a user-friendly Streamlit application.



---



## 📊 Dataset



The project uses the **IMDb Large Movie Review Dataset**.



The labeled dataset contains:



- 25,000 training reviews

- 25,000 testing reviews

- 50,000 labeled reviews in total

- Balanced positive and negative classes



The training data was divided into:



- 20,000 training reviews

- 5,000 validation reviews



The original 25,000-review test set was kept separate for final evaluation.



---



## 🧹 Data Preprocessing



The preprocessing pipeline includes:



- Text cleaning

- Removal of unnecessary HTML/text artifacts

- Tokenization

- Vocabulary construction for recurrent neural networks

- Sequence conversion

- Padding and truncation

- Transformer tokenization



For the LSTM and BiLSTM models, a vocabulary of 30,000 words was constructed.



For DistilBERT, the model's pretrained tokenizer was used with a maximum sequence length of 256 tokens.



---



## 🤖 Models Implemented



### 1. Logistic Regression



TF-IDF features were used as input to Logistic Regression.



**Test Accuracy: 89.95%**



---



### 2. Multinomial Naive Bayes



Multinomial Naive Bayes was trained using the TF-IDF representation.



**Test Accuracy: 86.83%**



---



### 3. Linear SVM



A Linear Support Vector Machine was trained using TF-IDF features.



**Test Accuracy: 89.11%**



---



### 4. LSTM



A Long Short-Term Memory neural network was implemented to capture sequential relationships between words.



Architecture included:



- Embedding layer

- LSTM layer

- Dropout

- Fully connected output layer



**Test Accuracy: 75.63%**



---



### 5. Bidirectional LSTM



A Bidirectional LSTM was implemented to process text sequences in both forward and backward directions.



This allows the model to incorporate contextual information from both directions.



**Test Accuracy: 77.07%**



---



### 6. DistilBERT



A pretrained DistilBERT transformer was fine-tuned for binary sentiment classification.



The model was trained for 3 epochs using GPU acceleration in Google Colab.



**Test Accuracy: 91.02%**



DistilBERT achieved the highest test accuracy among the evaluated models.



---



## 📈 Model Comparison



| Model | Test Accuracy |

|---|---:|

| DistilBERT | **91.02%** |

| Logistic Regression | 89.95% |

| Linear SVM | 89.11% |

| Multinomial Naive Bayes | 86.83% |

| BiLSTM | 77.07% |

| LSTM | 75.63% |



---



## 🔍 Error Analysis



The DistilBERT model was evaluated using a confusion matrix and classification report.



### DistilBERT Classification Results



| Class | Precision | Recall | F1-Score |

|---|---:|---:|---:|

| Negative | 0.9141 | 0.9056 | 0.9098 |

| Positive | 0.9065 | 0.9149 | 0.9107 |



Overall accuracy:



**91.02%**



The confusion matrix was:



```text

\[\[11320  1180]

&#x20;\[ 1064 11436]]


