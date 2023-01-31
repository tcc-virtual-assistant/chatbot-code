import json
from nltk_utils import tokenize, stem, bag_of_words
from unidecode import unidecode
import numpy as np

import torch 
from torch.utils.data import Dataset, DataLoader
from . import nn as nn

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

x_train = [] #pattern
y_train = [] #tag

for (pattern_tokenized, tag) in xy:
    bag = bag_of_words(pattern_tokenized, all_words)
    x_train.append(bag)
    
    tag_numbers = tags.index(tag) #identify with numbers instead the names
    y_train.append(tag_numbers)
x_train = np.array(x_train)
y_train = np.array(y_train)

#its necessary this class with this functions to create a custom dataset using pytorch
#so this is useful bc we can iterate automatically and better training
#initialize with all informations
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(x_train) #number of examples
        self.x_data = x_train  
        self.y_data = y_train

#returns a sample from the dataset at the given index
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
#returns the number of samples in our dataset
    def __len__(self):
        return self.n_samples
         
batch_size = 8
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)