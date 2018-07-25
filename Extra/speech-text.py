import speech_recognition as sr

# def listen():
#     r = sr.Recognizer()
#     print("Talk to J.A.R.V.I.S: ")
#     with sr.Microphone() as source:
#         audio = r.listen(source)
#         print(r.recognize_google(audio))

def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
    try:
        print(r.recognize_google(audio))
        return r.recognize_google(audio)
    except sr.UnknownValueError:
        speak(
            "I couldn't understand what you said! Would you like to repeat?")
        return(listen())
    except sr.RequestError as e:
        print("Could not request results from " + "Google Speech Recognition service; {0}".format(e))

listen()
