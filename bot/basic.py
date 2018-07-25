import re
from nltk import word_tokenize
import random

def listen():
	question=input("Human   :")
	return str(question)

def speak(reply):
	print("Chatbot :"+reply)

name=""
salutation="Sir"

#define all in raw_regex
# PENDING
raw_regex=\
{"greet":r"(\bhello\b)|(\bhi\b)|(\bhey\b)",
"bye":r"(\bbye\b)|(see you .*)"}
compiled_regex={}
for intent,exp in raw_regex.items():
	compiled=re.compile(exp)
	compiled_regex[intent]=compiled

#define all replies corresponding to an intent here
# PENDING
replies=\
{"greet":["Hello! Nice to see you here!","Hey! Welcome to our hotel."],
"bye":["Bye! It was a pleasure to have you in here.","Bye! Have a good day!"],
"default":["Sorry! I didn't get you.","Can you say that again please?"]}

def add_salutation(reply):
	if(len(re.findall("{}",reply))!=0):
		return reply.format(salutation)
	else:
		return reply

def pronoun_swap(question):
	#PENDING
	pronouns={"i":"you",\
			 "you":"i",\
			 "my":"your",\
			 "am":"are",\
			 "your":"my"}
	tokens = word_tokenize(question)
	for id,word in enumerate(tokens):
		if word in pronouns:
			tokens[id]=pronouns[word]
	return " ".join(tokens)

def get_intent(question):
	intent=""
	for cur_intent,regex in compiled_regex.items():
		arr= regex.findall(question)
		if(len(arr)!=0):
			intent=cur_intent
	return intent

def generate_reply(question):
	question=question.lower()
	intent=get_intent(question)
	if(intent in replies):
		reply=random.choice(replies[intent])
	else:
		reply=random.choice(replies["default"])
	return reply

def main():
	last_intent=""
	while(last_intent!="bye"):
		question=listen()
		last_intent=get_intent(question.lower())#if slow then optimise it 
		reply=generate_reply(question)
		speak(reply)

#main()
print(pronoun_swap("i am going to kill you!"))
print(add_salutation("Hello {}! Welcome to our hotel"))