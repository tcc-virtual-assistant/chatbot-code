import random
import json
# import train
import torch
import requests
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from sendUserAPI import sendAPI

def chatbot():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('intents.json', 'r', encoding="utf8") as json_data:
        intents = json.load(json_data)

    FILE = "data.pth"
    data = torch.load(FILE)

    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data['all_words']
    tags = data['tags']
    model_state = data["model_state"]

    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()
    print("Vamos testar! (Escreva 'sair' para sair)")

    allQuestions = requests.get('http://localhost:8000/question')
    allQuestions = (allQuestions.json())
    newest = 0
    for i in range(len(allQuestions)):
        if i > newest:
            newest = i
    newestQuestion = (allQuestions[newest]['userQuestion'])
    apiUserid = newest
    sentence = newestQuestion
    sentence = str(sentence)
    print(sentence)


    while True:
        if sentence == "sair":
            print( "Espero que eu tenha conseguido te ajudar! Tenha um ótimo dia!")
            break

        sentence = tokenize(str(sentence))
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.9993212223052980:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    id = intent['id']
            response = requests.get(f'http://localhost:5000/answer/{id}')
            call = response.json()
            answer = call['response']
            print(answer)
            sendAPI(answer, apiUserid)
            break
        else:
            print("Desculpe, eu não consegui compreender...")

chatbot()