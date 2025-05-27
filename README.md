# Customer Feedback Sentiment Analysis

This project performs sentiment analysis on customer reviews to generate actionable insights for a car rental agency, enabling data-driven decisions on inventory management.
![image](https://github.com/akhilesh360/Sentiment-Analysis-in-Dataiku-DSS/assets/70189275/cb8a3095-2a9b-46bc-b213-d719b4148db8)

**Key Objectives:**
- Extract and preprocess customer review text
- Train and evaluate a sentiment classification model
- Deploy the model to predict sentiment on new feedback

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Preprocessing](#preprocessing)
4. [Model Development](#model-development)
5. [Evaluation](#evaluation)
6. [Deployment](#deployment)
7. [Future Enhancements](#future-enhancements)
8. [License](#license)

---

## Project Overview
The goal is to analyze textual customer feedback and classify each review as Positive, Negative, or Neutral. Insights from sentiment predictions will help the agency optimize its car inventory and improve customer satisfaction.

**Workflow:**
1. **Data Collection**: Import raw review data containing car year, model, and review text.
2. **Text Preprocessing**: Normalize text, remove noise, and transform into features suitable for modeling.
3. **Model Training**: Train a logistic regression classifier on labeled data.
4. **Sentiment Prediction**: Apply the trained model to new reviews.
5. **Analysis & Reporting**: Summarize sentiment trends for business stakeholders.

## Dataset
- **Source:** Kaggle car review dataset
- **Fields:**
  - `Year`: Manufacturing year of the car
  - `Model`: Car model identifier
  - `Review`: Customer review text (input for sentiment analysis)
- **Train/Test Split:** 70% training, 30% testing

## Preprocessing
Text reviews undergo the following steps:
1. **Lowercasing & Punctuation Removal**: Convert text to lowercase and strip punctuation.
2. **Stop Word Removal**: Remove common words (e.g., "the", "is", "and").
3. **Tokenization & Stemming**: Split text into tokens and reduce words to their root forms.

```python
# Example preprocessing snippet
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))
review = "The car was GREAT! It exceeded my expectations."
# Lowercase & remove punctuation
cleaned = re.sub(r"[^a-zA-Z ]", "", review.lower())
# Tokenize & remove stop words
tokens = [word for word in cleaned.split() if word not in stop_words]
# Stemming
stemmed = [ps.stem(word) for word in tokens]
````

## Model Development

A logistic regression classifier is trained on preprocessed text features:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Feature extraction
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['Review'])
y = data['Sentiment']  # {Positive, Negative, Neutral}

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```

## Evaluation

Assess model performance using accuracy and confusion matrix metrics:

```python
from sklearn.metrics import accuracy_score, confusion_matrix

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
```


Find the step-by-step process done in my Dataiku DSS Environment,below : [SENTIMENT ANALYSIS PROJECT.pdf](https://github.com/akhilesh360/Sentiment-Analysis-in-Dataiku-DSS/blob/main/SENTIMENT%20ANALYSIS%20PROJECT.pdf)
Initial results:

* **Confusion Matrix:** Reveals distribution of correct and incorrect predictions by class.


## Future Enhancements

* Expand feature set with n-grams and sentiment lexicons
* Experiment with advanced models (e.g., BERT embeddings)
* Build a web interface for real-time review analysis

## License

This project is licensed under the MIT License. See the LICENSE file for details.

```

