import openpyxl
wb=openpyxl.Workbook()
sheet=wb.active
fileName="Ses05F_script02_2.txt"
fileName2="Ses05F_script02_2.xlsx"
f = open(fileName, "r")
fcolumn=('file_name','turn_name','start_time','end_time','text','emotion')
sheet.append(fcolumn)
i=1
r=1
while(i<68):
    asc=65
    data=f.readline()
    cellNo=chr(asc)
    fCNo=cellNo+str(r)
    asc+=1
    c2=data[:17]
    c3=data[18:22]
    c4=data[24:32]
    c5=data[33:41]
    c6=data[44:]
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
