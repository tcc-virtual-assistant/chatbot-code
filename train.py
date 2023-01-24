import json
from nltk_utils import tokenize, stem, bag_of_words
from unidecode import unidecode

with open('intents.json', 'r') as file:
    intents = json.load(file)

# colleting and tokenization
ignore_words = ['?', '!', ".", ","]
all_words = []
tags = []
xy = []
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        word = tokenize(pattern) #tokenize each word in phrase
        all_words.extend(word) #using extend instead append to not get array inside array
        xy.append((word, tag))

# stemming data
all_words = [stem(unidecode(word), '1') for word in all_words if word not in ignore_words]
all_words = sorted(set(all_words)) #ignore the words that has showed before
print(all_words)