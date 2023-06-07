import openpyxl
wb=openpyxl.Workbook()
sheet=wb.active

#Set search text here
searchText="fru"
fileName = "Dataset/Z_All_File.txt"
excelName = "Frustation_File"


excelName = excelName+".xlsx"


fTxt = open(fileName, "r")
fcolumn=('file_name','turn_name','start_time','end_time','emotion')
sheet.append(fcolumn)

while(True):
    asc=65
    data = fTxt.readline()
    if not data:
        break

    if "-" in data and "_" in data and "%" not in data:
        if searchText in data:
            sep=data.index("Ses")
            firstName=data[sep:sep+14]
            turnName=data[sep+15:sep+19]
            c1=data[:sep]
            startTime=data[c1.index("[")+1:c1.index("-")]
            endTime=data[c1.index("-")+1:c1.index("]")]
            emotion=data[sep+20:sep+23]
            ls=[]
            ls.append(firstName)
            ls.append(turnName)
            ls.append(startTime)
            ls.append(endTime)
            ls.append(emotion)
            tup=tuple(ls)
            sheet.append(tup)
        

    
wb.save(excelName)
print("Excel File Generated ")
            