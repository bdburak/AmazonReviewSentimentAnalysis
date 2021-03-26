#Review Seperator
def reviewToList(strDataLocation): #reviewToList(str_DataLocation)
    file = open(strDataLocation)
    listFile=(file.readlines())
    lastReviewItem=0
    firstReviewItem=0
    listReviews = []
    reviewText =""
    for item in range(len(listFile)):
        if('<review_text>\n'==listFile[item]):
            firstReviewItem = item+1
        if('</review_text>\n'==listFile[item]):
            ReviewItemRange = item - firstReviewItem
            for i in range(ReviewItemRange):
                reviewText = reviewText + (listFile[firstReviewItem])               
                firstReviewItem = firstReviewItem + 1
            listReviews.append(reviewText)
            reviewText =""
    file.close()
    return listReviews