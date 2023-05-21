import openpyxl
wb=openpyxl.Workbook()
sheet=wb.active

fname="Ses03M_impro01.txt"   
fname="Ses03M_impro01.txt"     #File Name
fileName=fname+".txt"
ExcelFile=fname+".xlsx"
fileNameEmo="emovalue/"+fileName
fileNameTxt="text/"+fileName


fTxt = open(fileNameTxt, "r")
fcolumn=('file_name','turn_name','start_time','end_time','text','emotion')
sheet.append(fcolumn)

i=1
r=1
emotion=""
while(True):
    asc=65
    data=fTxt.readline()
    if not data:
        break
    filenamedata=data[:19]
    cellNo=chr(asc)
    fCNo=cellNo+str(r)
    asc+=1
    c2=data[:14]
    c3=data[15:19]
    c4=data[21:29]
    c5=data[30:38]
    c6=data[41:]
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
                f1=edata[idx:idx+19]
                f2=edata[idx+20:idx+23]
                if(filenamedata==f1):
                    emotion=f2
        if not fEmo:
            break
    fEmo.close()
    ls.append(emotion)
    tup=tuple(ls)
    sheet.append(tup)
    i+=1
    r+=1
   
wb.save(ExcelFile)
print("Excel File Generated ")
