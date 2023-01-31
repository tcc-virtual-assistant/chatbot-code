import torch 
import torch.nn as nn   

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l1 = nn.Linear(input_size, num_classes)
        self.relu = nn.ReLU()
    

 
        