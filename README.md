# Spam Comment Identification using Naive Bayes Classifier

## Project Overview

This project focuses on training a Naive Bayes classifier to identify spam comments. The classifier is trained on a dataset containing comments from YouTube, specifically focusing on Shakira's videos.

## Project Structure

- `main.py`: Python script containing the code for data preprocessing, model training, and evaluation.
- `Youtube05-Shakira.csv`: Dataset file with comment data.
- `README.md`: Project documentation.

## Prerequisites

- Python
- Libraries: pandas, numpy, scikit-learn, nltk

## How to Run

1. Install the required libraries: `pip install pandas numpy scikit-learn nltk`.
2. Open and run the `spam_classifier.py` script in any Python IDE.

## Project Details

- Data Loading: The dataset is loaded and explored using pandas to understand its structure.
- Text Preprocessing: Comments are tokenized, stop words are removed, and stemming is applied.
- Vectorization: CountVectorizer is used to convert text data into a numerical format.
- Model Training: A Multinomial Naive Bayes classifier is trained on the transformed data.
- Cross-validation: The model is evaluated using 5-fold cross-validation.
- Testing: The trained model is tested on a set of new comments, and accuracy is reported.

Feel free to explore and modify the code to suit your needs!
