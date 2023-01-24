import nltk
from nltk.stem.porter import PorterStemmer

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
    pass