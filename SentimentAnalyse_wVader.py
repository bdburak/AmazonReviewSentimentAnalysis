from dataExtractor import reviewToList
from nltk.sentiment.vader import SentimentIntensityAnalyzer
model = SentimentIntensityAnalyzer()

"""IMPORTING FILES"""
tokenizedList=[]
reviewList = reviewToList("all.review") # electronics
reviewList_p = reviewToList("positive.review")
reviewList_n = reviewToList("negative.review")
positive_val = 0
positive_correct =0
negative_val = 0
negative_correct =0
neutr_val =0


for item in reviewList_p:
    scores_p = model.polarity_scores(item)
    if not scores_p['neg'] >0.1:
        if scores_p['pos']-scores_p['neg'] >=0:
            positive_correct +=1
    positive_val +=1

for item in reviewList_n:
    scores_n = model.polarity_scores(item)
    if not scores_n['pos'] >0.1:
        if scores_n['pos']-scores_n['neg'] <=0:
            negative_correct +=1
    negative_val +=1


print("VADER ACCURACY")
print("Positive accuracy = {}% via {} samples".format(positive_correct/positive_val*100.0,positive_val))
print("Negative accuracy = {}% via {} samples".format(negative_correct/negative_val*100.0,negative_val))
    
for item in reviewList:
    scores = model.polarity_scores(item)
    if(scores['compound']>0.1):
        positive_val +=1
    elif(scores['compound']<-0.1):
        negative_val +=1

print("VADER ANALYSE\n"+"Pos: "+ str(positive_val)+"\n"+ "Neg: "+ str(negative_val)+"\n"+"Neutr: "+ str(neutr_val)+"\n")



