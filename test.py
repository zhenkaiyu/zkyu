f = open("data/test_out.txt","r",encoding="utf-8")
f2 = open("data/result.txt","w",encoding="utf-8")
f3 = open("data/y.txt","r",encoding="utf-8")
f4 = open("data/y_pre.txt","r",encoding="utf-8")
f5 = open("data/as_predict9-10.utf8","r",encoding="utf-8")
# f6 = open("data/result.txt","r",encoding="utf-8")
text_set = []
y_correct = []
y_predict = []
b = 0
index_set = []
for line in f:
    text = line.strip().split()
    start = 0
    a = 0
    index = []
    for i in range(len(text)):
        word = text[i]
        if (word == "."or word == ";"):
            text_pre = text[start:i+1]
            text_set.append(text_pre)
            print(text_pre)
            start = i+1
            a = 1
            index.append(b)
            b = b+1
    if a == 1 and text[-1]!=".":
        for j in range(start,len(text)):
            text_set[-1].append(text[j])

    if a == 1:
        index_set.append(index)


    if a == 0:
        text_set.append(text)
        index.append(b)
        b = b+1
        index_set.append(index)
        print(text)

print(text_set)
# print(len(text_set))
for line in f3:
    if line.strip() == "1":
        y_correct.append("0")
    elif line.strip() == "0":
        y_correct.append("1")

for line in f4:
    if line.strip() == "1":
        y_predict.append("0")
    elif line.strip() == "0":
        y_predict.append("1")

# print(y_correct)
# print(y_predict)
# print(index_set)

# data_set_f1 = []
# temp_f1 = []
# for line in f5:
#     if len(line.strip()) == 0:
#         data_set_f1.append(temp_f1)
#         temp_f1 = []
#     else:
#         temp_f1.append(line.strip().split()[0])
#
# data_set_f= []
# temp_f = []
# for line in f6:
#     if len(line.strip()) == 0:
#         data_set_f.append(temp_f)
#         temp_f = []
#     else:
#         temp_f.append(line.strip().split()[0])
# print(data_set_f1)
# print(data_set_f)

#
#

for i in range(len(index_set)):
    index = index_set[i]
    for j in range(len(index)):
        a = index[j]
        text = text_set[a]
        y_cor = y_correct[a]
        y_pre = y_predict[a]
        for q in range(len(text)):
            word = text[q]
            f2.write(word)
            f2.write(" ")
            f2.write(y_pre)
            f2.write("\n")
    f2.write("\n")