
import re
from nltk import word_tokenize
import random
import spacy
import sqlite3
import os
import pyautogui
import time
import re
from PIL import Image
from pytesseract import image_to_string


def Sexyfunction():
    pyautogui.hotkey('alt', 'tab')
    time.sleep(2)
    im = pyautogui.screenshot('lunapic.jpg')
    pyautogui.hotkey('alt', 'tab')
    #print (image_to_string(Image.open('test.png')))
    return image_to_string(Image.open('lunapic.jpg'), lang='eng')

def match(name,string):
    if re.search(name.lower(),string.lower()):
        return True
    else :
        return False

print(Sexyfunction())
print(match("Kritin Garg",Sexyfunction()))
# b=input()
# printmatch("str","a b c str")