import os
import pandas as pd
import Fusion_Embed


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
            tamp=Fusion_Embed.ExtractFeature(temppath,'splitData\\imageSplit\\'+str(labels[i])+str(i)+'.wav')
        except:
            pass

          
from AudioFeature__ import ExtractFeatures

df = pd.DataFrame
sheet=pd.read_csv('dataset_label/P_GData.csv')
labels=sheet['emotion']
data=[]
path='audioSplit/'
for i in range(1236):
    if (i==0):
        continue
        
    temppath=path+'output'+str(i)+'.wav'
    print(temppath+'  '+str(labels[i-1]))
    Fusion_Embed=ExtractFeatures(temppath)

    Fusion_Embed.append(labels[i-1])
    data.append(Fusion_Embed)
data=pd.DataFrame(data)
data.to_csv('data2.csv')
