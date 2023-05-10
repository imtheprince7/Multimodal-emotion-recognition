import os
import pandas as pd
import temp

df = pd.DataFrame
sheet=pd.read_csv('dataSet\\dataset_label\\completeData.csv')
labels=sheet['emotion']
fileName=sheet['audioSplitFilename']
data=[]
path='splitData/audioSplit/'
for i in range(8980):
    if (i==0):continue
    if(os.path.isfile(path+str(fileName[i-1]))):
        try:
            # print(i)
            temppath=path+'output'+str(i)+'.wav'
            # print(temppath+'  '+str(labels[i-1]))
            tamp=temp.ExtractFeature(temppath,'splitData\\imageSplit\\'+str(labels[i])+str(i)+'.wav')
        except:
            pass

           