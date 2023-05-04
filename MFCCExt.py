import os
import pandas as pd
from AudioFeature__ import ExtractFeatures

df = pd.DataFrame
sheet=pd.read_csv('dataset_label/P_GData.csv')
labels=sheet['emotion']
data=[]
path='audioSplit/'
for i in range(1236):
    if (i==0):continue
    
    temppath=path+'output'+str(i)+'.wav'
    print(temppath+'  '+str(labels[i-1]))
    temp=ExtractFeatures(temppath)

    temp.append(labels[i-1])
    data.append(temp)
data=pd.DataFrame(data)
data.to_csv('data')

    
