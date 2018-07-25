import os
import shutil

i=0
inp="./AADHAR_IMAGES"
out="./NEW"
for img_name in os.listdir(inp):
	print(img_name)
	shutil.copy(inp+"/"+img_name,out+"/%d.jpg"%i)
	i+=1
