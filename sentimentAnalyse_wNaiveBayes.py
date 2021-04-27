from dataExtractor import reviewToList
from dataExtractor200 import dataExtractor200
import numpy as np
import random
"""IMPORTING FILES"""


reviewList_p = dataExtractor200("positive.review")
reviewList_n = dataExtractor200("negative.review")
X_pos_test = reviewList_p[1055:] 
X_neg_test = reviewList_n[665:]

"""Y_train and Y_test"""

Y_train = [] # 0 and 1
Y_test = []
for item in range(1720):
    if(item<=665):
        Y_train.append(0)
    else:
        Y_train.append(1)

#random.shuffle(Y_train)
print("Y_train shape :" +str(len(Y_train)))


for item in range(431):
    if(item<=166):
        Y_test.append(0)
    else:
        Y_test.append(1)
#random.shuffle(Y_test)
print("Y_test shape: "+str(len(Y_test)))
print(Y_test)
    

"""X_Train and X_test"""

X_train =[] 
pos_index = 0
neg_index =0
for item in range(len(Y_train)):
    if(Y_train[item]==0):
        X_train.append(reviewList_n[item-neg_index])
        pos_index +=1
    elif(Y_train[item]==1):
        X_train.append(reviewList_p[item-pos_index])
        neg_index +=1
print("X_train shape: "+str(len(X_train)))

X_test =[] 
X_neg_index = 0
X_pos_index =0
for item in range(len(Y_test)):
    if(Y_test[item]==0):
        X_test.append(X_neg_test[X_neg_index])
        X_neg_index +=1
    elif(Y_test[item]==1):
        X_test.append(X_pos_test[X_pos_index])
        X_pos_index +=1
print("Test Y and X--"+str(Y_test[0])+" : "+X_test[0])
print("Train Y and X--"+str(Y_train[0])+" : "+X_train[0])
print("X_test shape: "+str(len(X_test)))

if(len(X_test)!=len(Y_test) and len(X_train)!=len(Y_train)):
    print("X and Y shapes does not exists!")
else:
    print("Data hazırlama tamam \n --Data temizleme giriliyor.")



#----------------------------------------------------------------------------------

"""DATA CLEANING"""
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
import string
import re

tokenizer = RegexpTokenizer(r'\w+') # a-zA-Z0-9
en_stopwords = set(stopwords.words('english'))
en_stopwordsv2 = set(en_stopwords).union(set(ENGLISH_STOP_WORDS))
ps = PorterStemmer()

def getCleanedText(text):

    words = re.sub(r"[^A-Za-z\-]", " ",text).lower().split()
    tokens = [w for w in words if not w in en_stopwordsv2]
    lemmatizer = WordNetLemmatizer()
    cleaned_text = " ".join([lemmatizer.lemmatize(token) for token in tokens])
    return cleaned_text

X_train_clean = [getCleanedText(i) for i in X_train]
print(X_train_clean[0])

X_test_clean = [getCleanedText(i) for i in X_test]

print("Data temizleme tamam \n --Vektörize giriliyor.")

"""VECTORIZATION"""
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(ngram_range=(1,2),
                    max_df =0.5, 
                    min_df=10)
# N-gram olarak temizlenen metin vektörize edildi. Ve diziye aktarıldı.
X_vec = cv.fit_transform(X_train_clean).toarray() # eğitim verisi

X_test_vect = cv.transform(X_test_clean).toarray()

print("Data vektörize tamam \n --Sonuçlar gösterilecek.")
print("Makine öğrenme algoritması çalıştırılıyor.")

from sklearn.naive_bayes import MultinomialNB
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

mn = MultinomialNB()
mn.fit(X_vec,Y_train) # X_vec = Metinler, / Y_train = 0,1 lerden oluşan liste
print(X_vec.shape)
print(X_test_vect.shape)


Y_test_pred = mn.predict(X_test_vect)

print(Y_test_pred)

print("Naive bayes accuracy score : ",accuracy_score(Y_test_pred,Y_test)*100)

print(classification_report(Y_test,Y_test_pred))


cnf_matrix = confusion_matrix(Y_test,Y_test_pred)
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

labels = [0, 1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels)
plt.yticks(tick_marks, labels)
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
ax.xaxis.set_label_position("bottom")
plt.title('Confusion matrix')
plt.ylabel('Actual Value')
plt.xlabel('Predicted')
plt.show()


print("--------------------------------------------")
print("SVM çalışıyor.")

SVM = svm.SVC(C=0.6995,kernel='linear',gamma='auto')
SVM.fit(X_vec,Y_train)
predict_SVM = SVM.predict(X_test_vect)

print("SVM bitti.")
print(predict_SVM)
print("For linear kernel")
print("SVM accuracy score : ", accuracy_score(predict_SVM,Y_test)*100)
print(classification_report(Y_test,predict_SVM))

cnf_matrix = confusion_matrix(Y_test,predict_SVM)
import seaborn as sns
import pandas as pd

labels = [0, 1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels)
plt.yticks(tick_marks, labels)
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
ax.xaxis.set_label_position("bottom")
plt.title('Confusion matrix')
plt.ylabel('Actual Value')
plt.xlabel('Predicted')
plt.show()


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


models = []
models.append(('Logistic Regression', LogisticRegression(solver='liblinear',C=0.9,random_state=45)))
models.append(('RandomForestClassifier', RandomForestClassifier(random_state=45)))


for name, model in models:
    model = model.fit(X_vec, Y_train)
    y_pred = model.predict(X_test_vect)
    from sklearn import metrics

    print("%s -> ACC: %%%.2f" % (name,metrics.accuracy_score(Y_test, y_pred)*100))


# positive_correct = 0
# positive_val = 0
# negative_correct = 0
# negative_val = 0
# true_p =0
# false_p=0
# true_n=0
# false_n=0
# for item in range(len(X_test)):
#     if(Y_test_pred[item]==1 and Y_test[item]==1):
#         true_p+=1
#         positive_correct+=1
#         positive_val+=1
    
#     elif(Y_test_pred[item]==1 and Y_test[item]==0):
#         false_p+=1
#         negative_val+=1
#     elif(Y_test_pred[item]==0 and Y_test[item]==1):
#         false_n+=1
#         positive_val+=1
#     elif (Y_test_pred[item]==0 and Y_test[item]==0):
#         true_n+=1
#         negative_correct+=1
#         negative_val+=1

# totalCorrect = positive_correct+negative_correct
# totalVal = positive_val+negative_val

    
# precision_val = (true_p)/(true_p+false_p)
# recall_val = (true_p)/(true_p+false_n)
# f1Score_val = 2*((precision_val*recall_val)/(precision_val+recall_val))

# print("Accuracy = {}% via {} samples".format(totalCorrect/totalVal*100.0,totalVal))
# print("----------------")
# print("Accuracy = {}%".format(((true_p+true_n)/(true_p+false_p+true_n+false_n)*100.0)))
# print("----------------")
# print("Precision = {}".format(precision_val))
# print("----------------")
# print("Recall = {}".format(recall_val))
# print("----------------")
# print("F1 Score = {}".format(f1Score_val))
# print("Sonuçlar alındı") 

