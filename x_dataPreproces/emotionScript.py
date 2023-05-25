import openpyxl
wb=openpyxl.Workbook()
sheet=wb.active

fileName = "Ses01M_script01_3.txt"
excelName = "Ses01M_script01_3"

fileNameEmo="E:/Multimodal-emotion-recognition/data/emotion/"+fileName
fileNameTxt="E:/Multimodal-emotion-recognition/data/transcription/"+fileName

FileLocation="E:\\Multimodal-emotion-recognition\\datset_preProces\\dataset_Emotion\\"+excelName+".xlsx"


fTxt = open(fileNameTxt, "r")
fcolumn=('file_name','turn_name','start_time','end_time','text','emotion')
sheet.append(fcolumn)

i=1
r=1
x=1
emotion=""
while(True):
    asc=65
    data=fTxt.readline()
    if not data:
        break
    filenamedata=data[:22]
    cellNo=chr(asc)
    fCNo=cellNo+str(r)
    asc+=1
    c2=data[:17]
    c3=data[18:22]
    c4=data[24:31]
    c5=data[33:41]
    c6=data[44:]
    c6=c6[:len(c6)-1]
    ls=[]
    ls.append(c2)
    ls.append(c3)
    ls.append(c4)
    ls.append(c5)
    ls.append(c6)
    fEmo=open(fileNameEmo, "r")
    fEmo.readline()
    fEmo.readline()
    while(True):
        edata=fEmo.readline()
        if not edata:
            break
        if(len(edata)!=0):
            if(edata[0]=="["):
                idx=edata.index("Ses")
                f1=edata[idx:idx+22]
                f2=edata[idx+23:idx+26]
                if(filenamedata==f1):
                    emotion=f2
        if not fEmo:
            break
    fEmo.close()
    x+=1
    ls.append(emotion)
    tup=tuple(ls)
    sheet.append(tup)
    i+=1
    r+=1
   
wb.save(FileLocation)

print("File Generated: "+" "+excelName)