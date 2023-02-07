import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np

# nltk.download('rslp')
# nltk.download('punkt')

stemmer_eng = PorterStemmer()
stemmer_pt = nltk.stem.RSLPStemmer()

def tokenize(setence):
    return nltk.word_tokenize(setence)

def stem(word, idioma):
    if idioma == '1':
        return stemmer_pt.stem(word.lower())
    elif idioma == '2':
        return stemmer_eng.stem(word.lower())

def bag_of_words(tokenized_setences, all_words):
    #identify with boolean numbers the words
    #that are the same in the pattern to know what tag 
    #does it fit

    tokenized_setences = [stem(word, '1') for word in tokenized_setences]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for index, word in enumerate(all_words):
        if word in tokenized_setences:
            bag[index] = 1.0 
    return bag
