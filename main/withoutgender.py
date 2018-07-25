################################################################
#                    SEQ2SEQ MODELLING                         #
################################################################

from keras.layers import Input, Embedding, LSTM, Dense, RepeatVector, Dropout, merge
from keras.optimizers import Adam 
from keras.models import Model
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.preprocessing import sequence
from keras.layers import concatenate

import keras.backend as K
import numpy as np
np.random.seed(1234)  # for reproducibility
import _pickle as cPickle
import theano
import os.path
import sys
import nltk
import re
import time

from keras.utils import plot_model

word_embedding_size = 100
sentence_embedding_size = 300
dictionary_size = 7000
maxlen_input = 50

vocabulary_file = 'vocabulary_movie'
weights_file = 'my_model_weights20.h5'
unknown_token = 'something'
file_saved_context = 'saved_context'
file_saved_answer = 'saved_answer'
name_of_computer = 'Rebecca'

def greedy_decoder(input):

    flag = 0
    prob = 1
    ans_partial = np.zeros((1,maxlen_input))
    ans_partial[0, -1] = 2  #  the index of the symbol BOS (begin of sentence)
    for k in range(maxlen_input - 1):
        ye = model.predict([input, ans_partial])
        yel = ye[0,:]
        p = np.max(yel)
        mp = np.argmax(ye)
        ans_partial[0, 0:-1] = ans_partial[0, 1:]
        ans_partial[0, -1] = mp
        if mp == 3:  #  he index of the symbol EOS (end of sentence)
            flag = 1
        if flag == 0:    
            prob = prob * p
    text = ''
    for k in ans_partial[0]:
        k = k.astype(int)
        if k < (dictionary_size-2):
            w = vocabulary[k]
            text = text + w[0] + ' '
    return(text, prob)
    
    
def preprocess(raw_word, name):
    
    l1 = ['won’t','won\'t','wouldn’t','wouldn\'t','’m', '’re', '’ve', '’ll', '’s','’d', 'n’t', '\'m', '\'re', '\'ve', '\'ll', '\'s', '\'d', 'can\'t', 'n\'t', 'B: ', 'A: ', ',', ';', '.', '?', '!', ':', '. ?', ',   .', '. ,', 'EOS', 'BOS', 'eos', 'bos']
    l2 = ['will not','will not','would not','would not',' am', ' are', ' have', ' will', ' is', ' had', ' not', ' am', ' are', ' have', ' will', ' is', ' had', 'can not', ' not', '', '', ' ,', ' ;', ' .', ' ?', ' !', ' :', '? ', '.', ',', '', '', '', '']
    l3 = ['-', '_', ' *', ' /', '* ', '/ ', '\"', ' \\"', '\\ ', '--', '...', '. . .']
    l4 = ['jeffrey','fred','benjamin','paula','walter','rachel','andy','helen','harrington','kathy','ronnie','carl','annie','cole','ike','milo','cole','rick','johnny','loretta','cornelius','claire','romeo','casey','johnson','rudy','stanzi','cosgrove','wolfi','kevin','paulie','cindy','paulie','enzo','mikey','i\97','davis','jeffrey','norman','johnson','dolores','tom','brian','bruce','john','laurie','stella','dignan','elaine','jack','christ','george','frank','mary','amon','david','tom','joe','paul','sam','charlie','bob','marry','walter','james','jimmy','michael','rose','jim','peter','nick','eddie','johnny','jake','ted','mike','billy','louis','ed','jerry','alex','charles','tommy','bobby','betty','sid','dave','jeffrey','jeff','marty','richard','otis','gale','fred','bill','jones','smith','mickey']    

    raw_word = raw_word.lower()
    raw_word = raw_word.replace(', ' + name_of_computer, '')
    raw_word = raw_word.replace(name_of_computer + ' ,', '')

    for j, term in enumerate(l1):
        raw_word = raw_word.replace(term,l2[j])
        
    for term in l3:
        raw_word = raw_word.replace(term,' ')
    
    for term in l4:
        raw_word = raw_word.replace(', ' + term, ', ' + name)
        raw_word = raw_word.replace(' ' + term + ' ,' ,' ' + name + ' ,')
        raw_word = raw_word.replace('i am ' + term, 'i am ' + name_of_computer)
        raw_word = raw_word.replace('my name is' + term, 'my name is ' + name_of_computer)
    
    for j in range(30):
        raw_word = raw_word.replace('. .', '')
        raw_word = raw_word.replace('.  .', '')
        raw_word = raw_word.replace('..', '')
       
    for j in range(5):
        raw_word = raw_word.replace('  ', ' ')
        
    if raw_word[-1] !=  '!' and raw_word[-1] != '?' and raw_word[-1] != '.' and raw_word[-2:] !=  '! ' and raw_word[-2:] != '? ' and raw_word[-2:] != '. ':
        raw_word = raw_word + ' .'
    
    if raw_word == ' !' or raw_word == ' ?' or raw_word == ' .' or raw_word == ' ! ' or raw_word == ' ? ' or raw_word == ' . ':
        raw_word = 'what ?'
    
    if raw_word == '  .' or raw_word == ' .' or raw_word == '  . ':
        raw_word = 'i do not want to talk about it .'
      
    return raw_word

def tokenize(sentences):

    # Tokenizing the sentences into words:
    tokenized_sentences = nltk.word_tokenize(sentences)   #.decode(utf-8)
    index_to_word = [x[0] for x in vocabulary]
    word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
    tokenized_sentences = [w if w in word_to_index else unknown_token for w in tokenized_sentences]
    X = np.asarray([word_to_index[w] for w in tokenized_sentences])
    s = X.size
    Q = np.zeros((1,maxlen_input))
    if s < (maxlen_input + 1):
        Q[0,- s:] = X
    else:
        Q[0,:] = X[- maxlen_input:]
    
    return Q

 # Open files to save the conversation for further training:
qf = open(file_saved_context, 'w')
af = open(file_saved_answer, 'w')

# print('Starting the model...')

# *******************************************************************
#                Keras model of the chatbot: 
# *******************************************************************

ad = Adam(lr=0.00005) 

input_context = Input(shape=(maxlen_input,), dtype='int32', name='the_context_text')
input_answer = Input(shape=(maxlen_input,), dtype='int32', name='the_answer_text_up_to_the_current_token')
LSTM_encoder = LSTM(sentence_embedding_size, kernel_initializer= 'lecun_uniform', name='Encode_context')
LSTM_decoder = LSTM(sentence_embedding_size, kernel_initializer= 'lecun_uniform', name='Encode_answer_up_to_the_current_token')
if os.path.isfile(weights_file):
    Shared_Embedding = Embedding(output_dim=word_embedding_size, input_dim=dictionary_size, input_length=maxlen_input, name='Shared')
else:
    Shared_Embedding = Embedding(output_dim=word_embedding_size, input_dim=dictionary_size, weights=[embedding_matrix], input_length=maxlen_input, name='Shared')
word_embedding_context = Shared_Embedding(input_context)
context_embedding = LSTM_encoder(word_embedding_context)

word_embedding_answer = Shared_Embedding(input_answer)
answer_embedding = LSTM_decoder(word_embedding_answer)

merge_layer = concatenate([context_embedding, answer_embedding], axis=1, name='concatenate_the_embeddings_of_the_context_and_the_answer_up_to_current_token')
# print(dictionary_size)
out = Dense(dictionary_size//2, activation="relu", name='relu_activation')(merge_layer)
out = Dense(dictionary_size, activation="softmax", name='likelihood_of_the_current_token_using_softmax_activation')(out)

model = Model(inputs=[input_context, input_answer], outputs = [out])

model.compile(loss='categorical_crossentropy', optimizer=ad)

# plot_model(model, to_file='model_graph.png')    

if os.path.isfile(weights_file):
    model.load_weights(weights_file)


# Loading the data:
vocabulary = cPickle.load(open(vocabulary_file, 'rb'))

# print('computer: hi ! please type your name.\n')
# name = "Kushagra"
#Don't forget to initialize the name
# print('computer: hi , ' + name +' ! My name is ' + name_of_computer + '.\n') 

#takes input the name of the person also
def keras_reply(que,name):
    que = preprocess(que, name_of_computer)
    Q = tokenize(que)
    predout, prob = greedy_decoder(Q[0:1])
    start_index = predout.find('EOS')
    text = preprocess(predout[0:start_index], name)
    return text



#############################################
#                MAIN CODE                  #
#############################################


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



def listen():
	question=input("Human   :")
	return str(question)

def speak(reply):
	print("Chatbot :"+reply)


def entity(question):
	nlp = spacy.load('en')
	doc = nlp(question)
	name = ""
	for ent in doc.ents:
		if ent.label_ == "PERSON":
			name = ent.text
	return name 

salutation="Sir"

#define all in raw_regex
# PENDING

raw_regex=\
{"greet":r"(\bhello\b)|(\bhi\b)|(\bhey\b)",
"bye":r"(\bbye\b)|(\bsee you)",
"weather": r"(weather)",
"get_name":r"(\bname\b)|(\bcalled\b)|(i am)",
"room_type":r"(single)|(quad)|(twin)|(deluxe)",
"booked_online":r"(online)|(booked)",
"book_offline":r"(\bbook\b)|(offline)",
"thank":r"(thank)|(thanks)(thankyou)",
"room_service": r"(room service)|(intercom)|(emergency)|(contacts)",
"wifi": r"(wifi)",
"breakfast": r"(breakfast)",
"extra_bed": r"(extra bed)",
"atm": r"(\batm\b)",
"swimming": r"(swimming pool)|(pool)",
"driver": r"(driver)",
"tourist": r"(tourist)",
"restaurant": r"(restaurant)",
"checkout_timings" : r"(when can we check out)|(check out timings)|(timing for check out)|(timings for checkout)",
"check_out" : r"(i want to check out)|(i would like to check out)|(check out)",
"roomnumber" :r"[0-9]{1}|[0-9]{2}",
"aadhar_verification":r"(aadhar placed)",
"eigen_faces":r"(face picture)"
}

compiled_regex={}
for  intent,exp in raw_regex.items():
	compiled=re.compile(exp)
	compiled_regex[intent]=compiled

#define all replies corresponding to an intent here
# PENDING

replies=\
{"greet":["Hello! {} Nice to see you here! May I know your good name ?","Hey! {}  Welcome to our hotel. May I know your good name ?","Hello ! {} It is extremely good to see you on this fine and pleasant day! May I know your good name ? ",
        "Hello ! {} Good to see you .May I know your good name ?",
        "Hello! {} Nice to see you. May I know your good name ?"],
"bye":["Bye! {} It was a pleasure to have you in here.","Bye! {} Have a good day!"],
"get_name":["Hello! {} Nice to meet you! How can I help you? ","Pleased to meet you! {} How may I help you?","Welcome to our hotel! {} how may I help you?" ],
"booked_online": [" {} I need to check your name from our online database"],
"book_offline": ["Sure {} , We have with us a catalogue of our room prepared with us , please have a look over it and tell me which kind of room you would like to book"],
"default":["Sorry! {} I didn't get you.","Can you say that again please?"],
"thank":["Pleasure is always mine in serving customers like you {}. Have a nice day {} ."],
"room_service":["Room service is accessible by dialling 9 on the intercom. The whole list for all the numbers are shown up on your screen.",
"Just dial 9 on your intercom from your rooms. Further list is being shown.","It is 9 from your intercom, sir. Further list is being shown."],
"wifi":["For your floor it is 12345678"],
"breakfast":["The breakfast timings are 7.00am - 9.00am {}."],
"extra_bed":["An extra matress will cost 400 per night"],
"atm":["yes {} there is an ATM just down the lane.","the nearest ATM is down that lane just before the second right.","you can find an ATM near the second right on this road."],
"swimming":["the pool will be operational from 8 in the morning to 8 in the evening."],
"driver":["Sorry, there is no such provision at the hotel.","I'm afraid {} the driver would have to spend the night in the car it-self."],
"tourist":["Sure {}, we have a list of all such places prepared with us , you can have a look at that."],
"restaurant":["Sure {} . Actually I have a list prepared of all the nearby restaurants , here is the list."],
"checkout_timings":["The checkout timings are 11.00 AM."],
"check_out":["Sure {}, Could I have your room number please? Please place the room key the on the wall there."],
"roomnumber": ["Thanks {}, Hope you had a pleasant stay here, Please collect your credit card from the counter before leaving. Hope to see you again , Have a nice day {}."],
"weather": [" Today's weather is mostly assumed to remain clean {}" ]
}

def helper_res():
	file="restaurants.pdf"
	os.system("evince "+file)

def helper_room():
	file = "rooms.pdf"
	os.system("evince " + file)

def imp_con():
	file="contacts.pdf"
	os.system("evince "+file)

#           def aadhar_ocr(name):
# function defined for telling for the name matching output with aadhar

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
			 "your":"my",\
			 "me":"you"}
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

def generate_reply(questions,name=""):
    questions=questions.lower()
    intent=get_intent(questions)
    if(intent in replies):
        reply=random.choice(replies[intent])
    else:
        reply=keras_reply(questions,name)
    return reply


def find_room(params,name):
    conn =  sqlite3.connect('sqlite.db')
    c = conn.cursor()
    types , condition = params , "Vacant"
    t = (types , condition)
    c.execute('SELECT RoomNumber FROM HotelRooms WHERE RoomType = ? AND RoomStatus = ?',t)
    b = c.fetchone()
    if b == None:
        return "A room of that particulay type is not available {} , You may have a check in some other types of room."
    else :
        a = b[0]
        d = (name,a)
        # print(d)
        c.execute('UPDATE HotelRooms SET CustomerName = ?, RoomStatus = "Occupied" WHERE RoomNumber = ?',d)
        conn.commit()
        return a

def check_booking(name):
    conn = sqlite3.connect('sqlite.db')
    c = conn.cursor()
    t = (name,)
    c.execute(' SELECT RoomNumber FROM HotelRooms WHERE CustomerName = ? ',t)
    b = c.fetchone()
    if b == None:
        return "The name of the of the person is not found in online data"
    else :
        return b[0]


def check_out(room):
    conn = sqlite3.connect('sqlite.db')
    c = conn.cursor()
    d = (room,)
    c.execute(' SELECT RoomStatus FROM HotelRooms WHERE RoomNumber = ? ',d)
    v=  c.fetchone()
    e = v[0]  
    if e == "Vacant":
        return "Sorry {} I am afraid you need to check your room number once again, according to our database that room is vacant."
    else :
        c.execute('UPDATE HotelRooms SET CustomerName = "", RoomStatus = "Vacant" WHERE RoomNumber = ?',d)
        conn.commit()
        return "Thanks {} , Hope you had a pleasant stay here, Please collect your credit card from the counter before leaving. Hope to see you again , Have a nice day {}."


def get_room_number(question):
	b=[int(s) for s in question.split() if s.isdigit()]
	return b[0]


def online_booking(name):
	b = check_booking(name)
	if b == "The name of the of the person is not found in online data" :
		return "I am sorry {}, there is no room booking by your name. Check your name once , if you have written it correctly , if not then rewrite or else you have to contact the manager in case of any other query regarding it."
	else :
		return  "We have confirmed that you have a booking {} . But before alloting you the room we need to verify your aadhar .Will you please place your aadhar card in front the phone camera present here.Please type **Aadhar placed** here once image of your aadhar is being captured."


def aadhar(name):
	b = check_booking(name) 
	if match(name,Sexyfunction()) :
		return "We have one last thing left to be done , we have to verify by clicking a picture of your face .Once you are done with clicking picture type **face picture** over here "
	else:
		return "I am afraid {} that you need to go and talk to the manager for your room booking because your name does not match with on aadhar .For any other query please talk to the manager"

########### kushagra needs to fill in this function

from click_image import clicking
from eigenfaces import eigenfaces_main
#don't forget to add image location

def eigen_faces(name,img_location="./capture.jpg"):
    # return True
    clicking()
    a=eigenfaces_main("./faces",img_location)
    if(a==-1):
        return False
    else:
        return (name.lower()==a.lower())

def faces(name):
	b = check_booking(name)
	if eigen_faces(name):
		return "Thanks for your patience {} ,we have done all of our verification , your room number is " + str(b)
	else :
		return "Sorry I am afraid {} but your face doesn't matches with the data that we already had since your booking , I am extremely sorry but you have to see he manager."

def aadhar_ocr(name):t
		return True
############## kushagra needs to fill in this function

def offline_booking(params,name):
	b = find_room(params,name) 
	if b == "A room of that particular type is not available {} , You may have a check in some other types of room.":
		return "A room of that particulay type is not available {}, You can check in some other types of room but please mention completely which kind of room you want."
	else :
		return "Congratulations {}, there is availability of room and I have booked one for you. Your room number is "+ str(b)


def types(message):
    if (re.search("single", message) != None):
        return "single"
    elif (re.search("deluxe", message) != None):
        return "deluxe"
    elif (re.search("twin", message) != None):
        return "twin"
    elif (re.search("quad", message) != None):
        return "quad"

def gender_classifier():
    return "Sir"
#################################################################
#                  CHATBOT INTERFACE                            #
#################################################################
import tkinter as tk
try:
    import ttk as ttk
    import ScrolledText
except ImportError:
    import tkinter.ttk as ttk
    import tkinter.scrolledtext as ScrolledText
import time


class TkinterGUI(tk.Tk):

    def __init__(self, *args, **kwargs):
        """
        Create & set window variables.
        """
        tk.Tk.__init__(self, *args, **kwargs)

        self.chatbot = "" 
        self.salu = "Sir"
        self.number = 0
        self.title("RECEPTION CHATBOT")
        self.name=""
        self.room = 0
        self.params = ""
        self.initialize()

    def initialize(self):
        """
		Set window layout.
        """
        self.grid()

        self.respond = ttk.Button(self, text='Get Response', command=self.get_response)
        self.respond.grid(column=0, row=0, sticky='nesw', padx=3, pady=3)

        self.usr_input = ttk.Entry(self, state='normal')
        self.usr_input.grid(column=1, row=0, sticky='nesw', padx=3, pady=3)

        self.conversation_lbl = ttk.Label(self, anchor=tk.E, text='Conversation:')
        self.conversation_lbl.grid(column=0, row=1, sticky='nesw', padx=3, pady=3)

        self.conversation = ScrolledText.ScrolledText(self, state='disabled')
        self.conversation.grid(column=0, row=2, columnspan=2, sticky='nesw', padx=3, pady=3)
        
    def speak_response(self,user_input,reply):
        self.one = ttk.Label(self,text = "one")
        self.conversation['state'] = 'normal'
        self.conversation.insert(
        tk.END, "Human: " + user_input + "\n" + "ChatBot: " + str(reply) + "\n"     )
        self.conversation['state'] = 'disabled'
        time.sleep(0.5)


    def get_response(self):
        """
        Get a response from the chatbot and display it.
        """
        user_input = self.usr_input.get()
        self.usr_input.delete(0, tk.END)
        question=user_input
        reply=""
        self.salu = gender_classifier()
        last_intent=get_intent(question.lower())#if slow then optimise it
        if last_intent == "booked_online" :
            reply = online_booking(self.name)
            reply = re.sub("{}", self.salu ,reply)
            self.speak_response(question,reply)
        elif last_intent == "eigen_faces":
            reply = faces(self.name)
            reply = re.sub("{}", self.salu ,reply)
            self.speak_response(question,reply)
        elif last_intent == "get_name":
            self.number = 0
            self.name = entity(question)
            reply=generate_reply(question)
            reply = re.sub("{}", self.salu ,reply)
            self.speak_response(question,reply)
        elif last_intent == "restaurant":
            reply=generate_reply(question)
            reply = re.sub("{}", self.salu ,reply)
            self.speak_response(question,reply)
            helper_res()
        elif last_intent == "book_offline":
            reply=generate_reply(question)
            reply = re.sub("{}", self.salu ,reply)
            self.speak_response(question,reply)
            helper_room()
        elif last_intent == "room_service":
            reply=generate_reply(question)
            reply = re.sub("{}", self.salu ,reply)
            self.speak_response(question,reply)
            imp_con()
        elif last_intent == "aadhar_verification":
            reply = aadhar(self.name)
            reply = re.sub("{}", self.salu ,reply)
            self.speak_response(question,reply)
        elif last_intent == "room_type":
            self.params = types(question)
            reply = offline_booking(self.params,self.name)
            reply = re.sub("{}", self.salu ,reply)
            self.speak_response(question,reply)
        elif last_intent == "roomnumber":
            self.room = get_room_number(str(question))
            reply = check_out(self.room)
            reply = re.sub("{}", self.salu ,reply)
            self.speak_response(question,reply)
        elif self.number == 1:
            self.name = question
            self.number = 0
            reply = "Hello! {} Nice to meet you! How can I help you? "
            reply = re.sub("{}", self.salu ,reply)
            self.speak_response(question,reply)
        elif last_intent == "greet":
            self.number = 1
            reply=generate_reply(question)
            reply = re.sub("{}", self.salu ,reply)
            self.speak_response(question,reply)
        else :
            reply=generate_reply(question,self.name)
            reply = re.sub("{}", self.salu ,reply)
            self.speak_response(question,reply)



gui = TkinterGUI()
gui.mainloop()

# print(eigen_faces("Kushagra Juneja"))
