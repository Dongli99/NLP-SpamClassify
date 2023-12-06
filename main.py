# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 11:11:54 2023

@project: Identify spam comments by training an Naive Bayes classifier

@author: Dongli Liu
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load the data into a pandas data frame. 
data = pd.read_csv('Youtube05-Shakira.csv').iloc[:, -2:] # choose last 2 cols

# present results.  
print(data.shape)
print(data.columns)
print(data.dtypes)
print(data.describe())
print(data.isna().sum())
print(data.head())

# prepare train and test datasets
# shuffle data using pandas.sample
shuffled_data = data.sample(frac=1, random_state=38)

# tokenize(NOTE: tokenize and stop words are included in count_vectorizer)
# remove stop words and steming
tokenized_comments = [word_tokenize(text) for text in shuffled_data['CONTENT']]
stop_words = set(stopwords.words('English'))
filtered_comments = [[word for word in sentence if word.lower() not in stop_words]
                     for sentence in tokenized_comments]    
stemmer = PorterStemmer()
stemmed_comments = [[stemmer.stem(word)for word in sentence]
                     for sentence in filtered_comments]
# flatten the list
flattened_comments = [' '.join(sentence) for sentence in stemmed_comments] 
print(tokenized_comments[-1]) # display the results
print(filtered_comments[-1])
print(stemmed_comments[-1])
print(flattened_comments[-1])

# join the prepared content to the data
shuffled_data['CONTENT'] = flattened_comments

# split the shuffl
train_size = int(0.75 * len(shuffled_data))
trainX = shuffled_data[:train_size]['CONTENT']
trainY = shuffled_data[:train_size]['CLASS']
testX = shuffled_data[train_size:]['CONTENT']
testY= shuffled_data[train_size:]['CLASS']

# create and fit count_vectorizer
count_vectorizer = CountVectorizer()
trainX_vectorized = count_vectorizer.fit_transform(trainX)
print(trainX_vectorized)
print(trainX_vectorized.shape)

# Downscale the transformed data using tf-idf
tfidf = TfidfTransformer()
trainX_tfidf = tfidf.fit_transform(trainX_vectorized)
print(trainX_tfidf)
print(trainX_tfidf.shape)

# Fit the training data into a Naive Bayes classifier.  
classifier = MultinomialNB().fit(trainX_tfidf, trainY)

# Cross-validate the model on the training data using 5-fold
cv_scores = cross_val_score(classifier, trainX_tfidf, trainY, cv=5)
print(f'The score of the classifier are {cv_scores}')
print(f'Mean Accuracy:, {np.mean(cv_scores)}')

# transform test dataset
testX_vectorized = count_vectorizer.transform(testX)
testX_tfidf = tfidf.transform(testX_vectorized)
print(testX_tfidf)
# Test the model on the test data, print the confusion matrix and the  
# accuracy of the model.
predictions = classifier.predict(testX_tfidf)
conf_matrix = confusion_matrix(predictions, testY)
print('Confusion Matrix:')
print(conf_matrix)
accuracy = accuracy_score(predictions, testY)
print(f'The accuracy of the model:{accuracy}')

# new comments test data
comments = ['just for test I have to say murdev.com',
            'some of the best party to be had, full applause!!',
            'Can you please visit my store? just click this link',
            'I love this one❤❤❤❤',
            'Thank you very much for all the time sharing this '\
                'wonderful music 🎶 I wish you the best God bless you',
                'A me piace come cantante Shakira 😘'
            ]
# targets dataset
targets = [1,0,1,0,0,0]
# predict
predictions_new = classifier.predict(
    tfidf.transform(count_vectorizer.transform(comments)))
# evaluate prediction
conf_matrix_new = confusion_matrix(predictions_new, targets)
print('Confusion Matrix:')
print(conf_matrix_new)
accuracy_new = accuracy_score(predictions_new, targets)
print(f'The accuracy of the model:{accuracy_new}')