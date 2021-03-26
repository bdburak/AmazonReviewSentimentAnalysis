from textblob import TextBlob
from dataExtractor import reviewToList

pos_count=0
pos_correct=0

reviewList = reviewToList("all.review")# electronics
reviewList_p = reviewToList("positive.review")
reviewList_n = reviewToList("negative.review")



for item in reviewList_p:
    analysis = TextBlob(item)
    if(analysis.sentiment.polarity >=0.1):
        if(analysis.sentiment.polarity >= 0):
            pos_correct +=1
    pos_count +=1
neg_count =0
neg_correct=0

for item in reviewList_n:
    analysis = TextBlob(item)
    if(analysis.sentiment.polarity <0.1):
        if(analysis.sentiment.polarity < 0):
            neg_correct +=1
    neg_count +=1

print("TEXTBLOB ACCURACY")
print("Positive accuracy = {}% via {} samples".format(pos_correct/pos_count*100.0,pos_count))
print("Negative accuracy = {}% via {} samples".format(neg_correct/neg_count*100.0,neg_count))
sentimented = []
for item in reviewList:
        sentimented.append(TextBlob(item).sentiment.polarity)
neg_or_pos = []
positive=0
negative =0
notr =0
for i in range(len(sentimented)):
        if sentimented[i]<-0.1:
                neg_or_pos.append("Negative")
                negative +=1
        elif sentimented[i]>0.1:
                neg_or_pos.append("Positive")
                positive +=1
        

print("TEXTBLOB ANALYSE\n"+"Pozitif: "+str(positive)+"\nNegatif: "+str(negative)+"\nNötr: "+str(notr))

    