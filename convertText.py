import openpyxl
wb=openpyxl.Workbook()
sheet=wb.active
<<<<<<< HEAD
fileName="dataSet\\text\\Ses01F_impro01.txt"
fileName2="Ses01F_impro01.xlsx"
=======
fileName="E:/Multimodal-emotion-recognition/text/Ses05M_impro01.txt"
fileName2="Ses05M_impro01.xlsx"
>>>>>>> fa1822ec888780ef2140ea7e16718fd2d9b70d4b
f = open(fileName, "r")
fcolumn=('file_name','turn_name','start_time','end_time','text','emotion')
sheet.append(fcolumn)
i=1
r=1
<<<<<<< HEAD

=======
>>>>>>> fa1822ec888780ef2140ea7e16718fd2d9b70d4b
while(i<100):
    asc=65
    data=f.readline()
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
    tup=tuple(ls)
    sheet.append(tup)
    i+=1
    r+=1

wb.save(fileName2)
<<<<<<< HEAD
print("Excel File Generated ")
=======
>>>>>>> fa1822ec888780ef2140ea7e16718fd2d9b70d4b
