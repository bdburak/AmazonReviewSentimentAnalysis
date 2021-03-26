from dataExtractor import reviewToList
import numpy as np
import random
"""IMPORTING FILES"""
reviewList = reviewToList("all.review")# electronics
reviewList_p = reviewToList("positive.review")
reviewList_n = reviewToList("negative.review")


"""Y_train"""
zeros_t = np.zeros(800)
ones_t = np.ones(800)
train_dataset = [] # 0 and 1 
for item in range(1600):
    if(item%2 ==0):
        train_dataset.append(0)
    else:
        train_dataset.append(1)
random.shuffle(train_dataset)

train_text =[] # text train_text dataset %80 of all texts (1600)
pos_index = 0
neg_index =0
for item in range(len(train_dataset)):
    if(train_dataset[item]==0):
        train_text.append(reviewList_n[item-neg_index])
        pos_index +=1
    elif(train_dataset[item]==1):
        train_text.append(reviewList_p[item-pos_index])
        neg_index +=1



pos_test_text = reviewList_p[800:] # test_text dataset %20 of all texts (400)
neg_test_text = reviewList_n[800:]
all_test_text = reviewList


"""DATA CLEANING"""
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

tokenizer = RegexpTokenizer(r'\w+')
en_stopwords = set(stopwords.words('english'))
ps = PorterStemmer()

def getCleanedText(text):
    text = text.lower()

    tokens = tokenizer.tokenize(text)
    new_tokens = [token for token in tokens if token not in en_stopwords]

    stemmed_tokens = [ps.stem(tokens) for tokens in new_tokens]

    cleaned_text = " ".join(stemmed_tokens)

    return cleaned_text

X_train_clean = [getCleanedText(i) for i in train_text]
pos_test_clean = [getCleanedText(i) for i in pos_test_text]
neg_test_clean = [getCleanedText(i) for i in neg_test_text]
all_test_clean = [getCleanedText(i) for i in all_test_text]


"""VECTORIZATION"""
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(ngram_range=(1,2))

X_vec = cv.fit_transform(X_train_clean).toarray()

pos_test_vect = cv.transform(pos_test_clean).toarray()
neg_test_vect = cv.transform(neg_test_clean).toarray()
all_test_vect = cv.transform(all_test_clean).toarray()



"""MULTINOMINAL NAIVE BAYES"""

from sklearn.naive_bayes import MultinomialNB

mn = MultinomialNB()

mn.fit(X_vec,train_dataset)

pos_y_pred = mn.predict(pos_test_vect)
neg_y_pred = mn.predict(neg_test_vect)
all_y_pred = mn.predict(all_test_vect)


"""TESTING SECT"""

positive_correct = 0
positive_val = 0
negative_correct = 0
negative_val = 0
for item in range(len(pos_test_text)):
    if(pos_y_pred[item]==1):
        positive_correct +=1
    positive_val +=1
for item in range(len(neg_test_text)):
    if(neg_y_pred[item]==0):
        negative_correct +=1
    negative_val +=1

print("Positive accuracy = {}% via {} samples".format(positive_correct/positive_val*100.0,positive_val))
print("Negative accuracy = {}% via {} samples".format(negative_correct/negative_val*100.0,negative_val))

"""ML ANALYSE""" 
positive_val =0
negative_val =0
for item in range(len(reviewList)):
    if(all_y_pred[item]==1):
        positive_val+=1
    elif(all_y_pred[item]==0):
        negative_val+=1

print("NAIVE BAYES\n"+"Pos: "+ str(positive_val)+"\n"+ "Neg: "+ str(negative_val))
    