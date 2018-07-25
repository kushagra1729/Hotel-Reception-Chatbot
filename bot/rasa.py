from rasa_nlu.training_data import load_data
from rasa_nlu.model import Metadata, Interpreter
message = u"I want to book a flight to London"
Interpreter.parse(message)
