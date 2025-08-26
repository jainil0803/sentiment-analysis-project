# sentiment-analysis-project
Amazon Reviews Sentiment Analysis
This project focuses on analyzing customer reviews from the Amazon Fine Food Reviews dataset (Reviews.csv). The main goal is to preprocess, clean, and build machine learning / deep learning models to classify reviews based on their sentiment.
📂 Project Structure
📦 sentiment-analysis
 ┣ 📜 Reviews.csv         # Dataset
 ┣ 📜 notebook.ipynb      # Jupyter Notebook with code
 ┣ 📜 README.md           # Project documentation
 ┣ 📜 requirements.txt    # Dependencies
 ┗ 📂 models/             # Saved trained models
🚀 Getting Started
1. Clone the repository
git clone https://github.com/yourusername/sentiment-analysis.git
cd sentiment-analysis
2. Install dependencies
pip install -r requirements.txt
3. Run Jupyter Notebook
jupyter notebook
📊 Dataset
File: Reviews.csv
Columns:
- Id → Review ID
- ProductId → Unique product identifier
- UserId → Reviewer’s ID
- ProfileName → Reviewer’s name
- HelpfulnessNumerator / HelpfulnessDenominator → Helpfulness score
- Score → Rating (1–5 stars)
- Time → Timestamp of review
- Summary → Short text summary of review
🧹 Data Cleaning
1. Removed missing values and duplicates
2. Converted timestamps to readable dates
3. Lowercased and tokenized text data
4. Removed stopwords, punctuation, and special characters
🤖 Models Used
- Logistic Regression (Baseline)
- Naive Bayes
- LSTM / RNN (Deep Learning)
- Transformers (BERT-based, optional)
📈 Evaluation
Metrics: Accuracy, Precision, Recall, F1-score
Visualizations: Confusion Matrix, ROC Curve
📌 Results
- Logistic Regression baseline achieved ~XX% accuracy
- Deep Learning models improved results significantly
🔮 Future Work
- Improve preprocessing with lemmatization/stemming
- Try advanced transformer-based models (BERT, DistilBERT)
- Deploy as a web app (Flask / FastAPI + Streamlit for UI)
🙌 Acknowledgements
- Dataset: Amazon Fine Food Reviews (Kaggle)
- Libraries: Pandas, Scikit-learn, TensorFlow/Keras, NLTK, Matplotlib
