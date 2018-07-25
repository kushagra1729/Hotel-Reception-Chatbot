#!/bin/bash                                                                     
#original images in ./old/                                                      
#resized images in ./new/                                                       
#labels in ./labels.txt                                                         
#Nikhil, 2018 jun 13 22:30 +530                                                 
#[ -d "./new" ] || mkdir new
j=0
for i in $(ls AADHAR_IMAGES)
do
	str1=NEW/
	str2=$j
	str3=.jpg
	str4="$str1$str2$str3"
    cp AADHAR_IMAGES/$i str4
    ((++j))
done