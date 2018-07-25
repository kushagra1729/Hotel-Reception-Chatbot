file=open("answers_simple.txt","r")
content=file.read()
strings=content.split('\n')
# dialogues=dialogues[1:]
# strings=[]

# def improve(string):
# 	j=0
# 	# print(string)
# 	while(string[j:]!="" and string[j]==' '):
# 		j+=1
# 	string=string[j:]
# 	if(string!=""):
# 		j=-1
# 		# print(string)
# 		while(string[j]==' '):
# 			j-=1
# 		j+=1
# 		if(j<0):
# 			string=string[:j]
# 		# if(string[0].lower()==string[0]):
# 		# 	# print(string)
# 		# print(string)
# 		strings.append(string)

# for i,st in enumerate(dialogues):
# 	arr=dialogues[i].split(':')
# 	for line in arr:
# 		improve(line)
# 	# print(dialogues[0:5])

for i,string in enumerate(strings):
	if(i!=len(strings)-1):
		print(strings[i],strings[i+1])
# 		
