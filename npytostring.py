f = open("data/9-10test.txt","r",encoding="utf-8")
f1 = open("data/aaa.txt","w",encoding="utf-8")
line = f.readline()
text = line.strip().split()
print(text)
for i in range(len(text)):
    word = text[i]
    f1.write(word)
    f1.write(" ")
    f1.write("1")
    f1.write("\n")