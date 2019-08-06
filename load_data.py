# import openpyxl
#
# wb = openpyxl.load_workbook("data/sentimen.xlsx")
#
# sheetname = wb.sheetnames[0]
# column = 'C'
# column1 = 'D'
# table = wb[sheetname]
# cell = table[column]
# cell1 = table[column1]
# label = []
# text = []
# num = 0
# for c in cell:
#     num = num+1
#     if c.value!=None:
#         label.append(c.value)
#     if c.value == None and num<=6000:
#         print(num)
# for c in cell1:
#     if c.value!=None:
#         text.append(c.value)
#
# # print(label)
# # print(text)
# f = open("data/senti_label.txt","w",encoding="utf-8")
# f1 = open("data/senti_text.txt","w",encoding="utf-8")
# for i in range(len(label)):
#     f.write(str(label[i]))
#     f.write("\n")
# for i in range(0,6000):
#     f1.write(text[i])
#     f1.write("\n")

f = open("data1/senti_text_cut.txt","r",encoding="utf-8")
f1 = open("data1/senti_label.txt","r",encoding="utf-8")
f2 = open("data1/senti_data.txt","w",encoding="utf-8")

for line1,line2 in zip(f,f1):
    f2.write(line2.strip()+" "+line1)

