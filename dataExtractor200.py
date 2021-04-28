from dataExtractor import reviewToList

def count_chars(txt):
    res =0
    for chr in txt:
        res+=1
    return res      
#print(count_chars('Test'))
def dataExtractor200(strDataLocation):
    reviewList = reviewToList(strDataLocation)
    features = [] 
    for item in reviewList:
        count = count_chars(item)
        #print(count)
        if(count<=200):
            features.append(item)
            count = 0
        else:
            count= 0
    return features
