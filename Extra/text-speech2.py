# import os
# import time
# # import pyttsx
# from gtts import gTTS
# from pygame import mixer

# # def gtts_speak(jarvis_speech):
# # 	tts = gTTS(text=jarvis_speech, lang='en')
# # 	tts.save('jarvis_speech.mp3')
# # 	mixer.init()
# # 	mixer.music.load('jarvis_speech.mp3')
# # 	mixer.music.play()
# # 	while mixer.music.get_busy():
# # 		time.sleep(1)

# # def offline_speak(jarvis_speech):
# # 	engine = pyttsx.init()
# # 	engine.say(jarvis_speech)
# # 	engine.runAndWait()

# import pyttsx3
# def offline_speak(jarvis_speech):
# 	engine = pyttsx3.init()
# 	engine.say(jarvis_speech)
# 	engine.setProperty('rate',120)  #120 words per minute
# 	engine.setProperty('volume',0.9) 
# 	engine.runAndWait()

# #gtts_speak("hello world")
# offline_speak("hello world")

import pyttsx3;
engine = pyttsx3.init();
engine.say("I will speak this text");
engine.runAndWait() ;