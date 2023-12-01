# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 11:11:54 2023

@project:

@authors: 
    Dongli Liu 301268638
    
    (Please add your name and student ID)
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from nltk import stem

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
stemmer = stem.PorterStemmer()
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
trainX_vct = count_vectorizer.fit_transform(trainX)
print(trainX_vct)
print(trainX_vct.shape)

# Downscale the transformed data using tf-idf
tfidf = TfidfTransformer()
trainX_tfidf = tfidf.fit_transform(trainX_vct)
print(trainX_tfidf)
print(trainX_tfidf.shape)

# Fit the training data into a Naive Bayes classifier.  
classifier = MultinomialNB()




# 9. Cross validate the model on the training data using 5-fold and print the 
# mean results of model accuracy. 

# 10. Test the model on the test data, print the confusion matrix and the  
# accuracy of the model. 

# 11. As a group come up with 6 new comments (4 comments should be non spam and  
# 2 comment spam) and pass them to the classifier and check the results. You  
# can be very creative and even do more Shape happy with light skin tone 
# emoticon. 

# 12. Present all the results and conclusions. 

# 13. Drop code, report and power point presentation into the project 
# assessment folder for grading. 