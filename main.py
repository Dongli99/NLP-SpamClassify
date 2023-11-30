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
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
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

# prepare corpus
contents = data['CONTENT']
classes = data['CLASS']

# tokenize(NOTE: tokenize and stop words are included in count_vectorizer)
# remove stop words and steming
tokenized_contents = [word_tokenize(text) for text in contents]
stop_words = set(stopwords.words('English'))
filtered_contents = [[word for word in sentence if word.lower() not in stop_words]
                     for sentence in tokenized_contents]    
stemmer = stem.PorterStemmer()
stemmed_contents = [[stemmer.stem(word)for word in sentence]
                     for sentence in filtered_contents]
# 
flattened_contents = [' '.join(sentence) for sentence in stemmed_contents] 
print(tokenized_contents[-7]) # display the results
print(filtered_contents[-7])
print(stemmed_contents[-7])
print(flattened_contents[-7])

# create count_vectorizer
count_vectorizer = CountVectorizer()
training_corpus=count_vectorizer.fit_transform(flattened_contents)

# 4. Present highlights of the output (initial features) such as the new shape 
# of the data and any other useful information before proceeding. 
print(training_corpus.shape)


# 5. Downscale the transformed data using tf-idf and again present highlights 
# of the output (final features) such as the new shape of the data and any  
# other useful information before proceeding. 

# 6. Use pandas.sample to shuffle the dataset, set frac =1  

# 7. Using pandas split your dataset into 75% for training and 25% for testing, 
# make sure to separate the class from the feature(s). 
# (Do not use test_train_ split) 

# 8. Fit the training data into a Naive Bayes classifier.  

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