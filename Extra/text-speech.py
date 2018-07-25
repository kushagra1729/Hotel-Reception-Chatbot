from gtts import gTTS
import os
tts = gTTS(text='Kritin is chutiya', lang='en')
tts.save("good.mp3")
os.system("mpg321 good.mp3")
tts2 = gTTS(text='Yes, Kritin is maha chutiya', lang='en')
tts2.save("good.mp3")
os.system("mpg321 good.mp3")

