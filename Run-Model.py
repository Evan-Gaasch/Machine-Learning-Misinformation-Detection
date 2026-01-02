print("initializing, please wait")
import torch
from torch import nn
import numpy as np
import pandas as pd
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import GloVe

tokenizer = get_tokenizer("basic_english")

#pretrained vocabulary
glove = GloVe(name='6B', dim=100)
vocab = glove.stoi

def toTensor(input_text):
    #tokenize and convert to tensor
    token_ids = []
    tokens = tokenizer(input_text)
    #print(tokens)
    for i in range(len(tokens)):
        if tokens[i] in vocab:
            token_ids.append(vocab[tokens[i]]) #list of integers of words in the order they appear for each example
        else:
            token_ids.append(0)
            #print("unkown")

    #print(token_ids)
    token_ids = torch.tensor(token_ids,dtype = torch.float32)
    #print(token_ids.size())
    
    #ensure constant tensor length by padding tensor length so it is always 512
    pad_dimension = (0,512) #will trim later
    token_ids = F.pad(token_ids,pad_dimension, "constant",0)
    #print(token_ids.size())
    #trim token_ids to first 512 characters
    token_ids = token_ids[0:512]
    token_ids = token_ids.unsqueeze(0) #add a second dimension
    return token_ids

#set up network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(512, 900), #input
            nn.ReLU(),
            #nn.Dropout(0.1),
            nn.Linear(900, 1024), #hidden 1 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1200), #hidden 2
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1200, 1600), #hidden 3
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1600, 1900), #hidden 4
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1900, 2048), #hidden 5
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 3500), #hidden 6
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(3500, 4096), #hidden 7
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 3500), #hidden 8
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(3500, 2048), #hidden 9
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1900), #hidden 10
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1900, 1600), #hidden 11
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1600, 1200), #hidden 12
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1200, 1024), #hidden 13
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 900), #hidden 14
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(900, 700), #hidden 15
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(700, 512), #hidden 16
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 300), #hidden 17
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(300, 200), #hidden 18
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(200, 150), #hidden 19
            nn.ReLU(),
            nn.Linear(150, 100), #hidden 20
            nn.ReLU(),
            nn.Linear(100, 2),#output
       )
   

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)

        return logits

#load model
model = NeuralNetwork().to("cpu")#if using colab and cuda gpu, switch to "cuda"
model.load_state_dict(torch.load('./model_weights.pth'))
print("loaded model")
model.eval()
with torch.no_grad():
    correct = 0
    counter = 0
    real = 0
    fake = 0
    false_detected_incorrectly = 0 #was fake said real
    false_detected_correctly = 0 #was fake said fake
    true_detected_incorrectly = 0 #was real, said fake
    true_detected_correctly = 0 #was real said real

    while True:
        counter += 1
        print("Enter possible misinformation here:")
        possible_misinformation = input(">")
        possible_misinformation = toTensor(possible_misinformation)
        possible_misinformation = possible_misinformation.to("cpu")
        prediction = model(possible_misinformation)
        print(prediction)
        _, predicted = torch.max(prediction,1)
        
        predicted = predicted.tolist()
        print(predicted)
        #calculate probabilities
        chance = F.softmax(prediction, dim=1).tolist()
        #chance = 
        if predicted == [0]:
            #chance = (prediction[0,0]/(prediction[0,0]+prediction[0,1])).tolist()
            print(chance)
            chance = float(chance[0][0])
            chance *= 100
            chance = str(chance)
            print("likely accurate with a " +chance+ "% chance")
            if input("do you agree (y/n)") == "y":
                correct += 1
                real += 1
                true_detected_correctly += 1

            else:
                fake += 1
                false_detected_incorrectly += 1
                
            
        else:
            #chance = (prediction[0,1]/(prediction[0,0]+prediction[0,1])).tolist()
            print(chance)
            chance = float(chance[0][1])
            chance *= 100
            chance = str(chance)
            
            print("likely false with a " +chance+ "% chance")
            if input("do you agree (y/n)") == "y":
                correct += 1
                fake += 1
                false_detected_correctly += 1
                
            else:
                real += 1
                true_detected_incorrectly += 1
        
        #diagnostics for tuning
        print("accuracy:"+str((correct/counter)*100)+"%")
        print("Real: "+ str(real) + " Fake: "+ str(fake))
        print("false_detected_correctly: "+str(false_detected_correctly)+ " false_detected_incorrectly: " + str(false_detected_incorrectly))
        print("true_detected_correctly: "+str(true_detected_correctly)+ " true_detected_incorrectly: " + str(true_detected_incorrectly))
