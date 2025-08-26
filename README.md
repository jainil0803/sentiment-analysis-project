# sentiment-analysis-project


# 📝 Sentiment Analysis with Deep Learning

This project applies Natural Language Processing (NLP) and Deep Learning to classify customer reviews into **positive** or **negative** sentiments.  
It leverages TensorFlow/Keras models with text vectorization for robust text classification.

---

## 📊 Dataset Overview

- **Source**: `Reviews.csv`  
- Contains customer reviews with sentiment labels (Positive / Negative).  
- First few rows are loaded with `pandas.read_csv()` for exploration.

---

## 🧪 Exploratory Data Analysis (EDA)

### Key Steps:
1. **Data Inspection**  
   - Checked dataset structure using `df.head()` and `df.info()`.
   - Verified class distribution of sentiments.  

2. **Visualizations**  
   - Plotted sentiment counts.  
   - Generated confusion matrix for model evaluation.  

---

## 🔧 Preprocessing

1. **Text Vectorization**  
   - Used `TextVectorization` layer from `tensorflow.keras.layers` to convert raw text into integer sequences.  

2. **Train-Test Split**  
   - Data split into training and testing sets using `train_test_split`.  

3. **Normalization**  
   - Converted all text to lowercase.  
   - Removed punctuation/special characters automatically via vectorization.  

---

## 🧠 Modeling

- Implemented a **Sequential Neural Network** using TensorFlow/Keras:  
  - Input layer: Text vectorization  
  - Hidden layers: Dense layers with activation functions  
  - Output layer: Sigmoid for binary classification  

### Training  
- Trained with `model.fit()` for **10 epochs**.  
- Monitored performance using **validation accuracy** and **loss curves**.  

---

## 📈 Evaluation

- **Confusion Matrix**: Visualized using `seaborn.heatmap`.  
- **Metrics**: Accuracy, precision, recall, F1-score.  
- Example confusion matrix:  
  - High accuracy on both Positive and Negative reviews.  

---

## 🔮 Predictions

- Tested model with new unseen reviews:  
  - `"This was the best purchase I have ever made, absolutely fantastic!"` → **Positive**  
  - `"A complete waste of money. I would not recommend this product at all."` → **Negative**  
  - `"The movie was just okay, not great but not terrible either."` → **Neutral-ish (classified accordingly)**  

---

## 🌟 Insights and Learnings

- Deep learning models can effectively classify customer sentiments with high accuracy.  
- Text vectorization simplifies preprocessing compared to manual tokenization.  
- Confusion matrix highlights that some neutral reviews are harder to classify.  

---

## 🛠️ Tools and Technologies

- **Libraries**:  
  - `pandas`, `numpy`, `matplotlib`, `seaborn`  
  - `sklearn` (train-test split, metrics)  
  - `tensorflow`, `keras` (deep learning)  

---

## 🖋️ Conclusion

This project demonstrates how **deep learning and NLP** can be used to automatically classify sentiment in reviews.  
It provides a baseline that can be improved further with advanced embeddings (Word2Vec, BERT) and larger datasets.

---

Feel free to fork this repo ⭐, run the notebook, and improve the model. Contributions are always welcome!
