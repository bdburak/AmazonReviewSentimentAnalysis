from dataExtractor import reviewToList
import csv

reviewListN = reviewToList("negative.review")
reviewListP = reviewToList("positive.review")

with open('techNew.tsv', 'wt') as out_file:
    
    for item in range(min(len(reviewListN),len(reviewListP))):    
        tsv_writer = csv.writer(out_file, delimiter='\t')
        
        reviewListP[item] = reviewListP[item].rstrip("\n")
        reviewListN[item] = reviewListN[item].rstrip("\n")
        if(len(reviewListP[item]) <= 200):
            
            tsv_writer.writerow([reviewListP[item], 1])
        #[reviewListN[item][0:len(reviewListN[item])-1]
        if(len(reviewListN[item]) <= 200):
            tsv_writer.writerow([reviewListN[item], 0])
    


#burda sirayla ikisindi de birlesitirmem ve csv olarak yazmam lazim
#https://riptutorial.com/python/example/26946/writing-a-tsv-file