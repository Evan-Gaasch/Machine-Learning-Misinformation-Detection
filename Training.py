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
import signal 

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
    #print(token_ids)
    return token_ids


def signal_handler(sig, frame):
    torch.save(model.state_dict(), 'E:/Evan/Coding/code/AI/model_architecture/model_weights.pth')
    print('saved, exiting')
    exit(0)
    
    
#optimization settings (tune)
learning_rate = 0.0001 
batch_size =40
epochs = 10000


#set up dataset
truth_dataset = pd.read_csv("E:/Evan/Coding/code/AI/Misinformation_Dataset/True.csv")
lie_dataset = pd.read_csv("E:/Evan/Coding/code/AI/Misinformation_Dataset/Fake.csv") #creates datasets
real_dataset = pd.read_csv("E:/Evan/Coding/code/AI/Misinformation_Dataset/real_world.csv")
print(truth_dataset)

#set up second dataset
dataset_2 = pd.read_csv("E:/Evan/Coding/code/AI/Misinformation_Dataset/real.csv")
print(dataset_2)

#extract just "text" column, what we care about.
truth_data = truth_dataset[["text"]]
lie_data = lie_dataset[["text"]]
print(truth_data)
print(lie_data)
#annotate datasets
truth_data = truth_data.assign(Category=0) #no misinformation
#print(truth_data)
lie_data = lie_data.assign(Category = 1) #misinformation



#combine datasets
combined_data = pd.concat([truth_data,lie_data], axis = 0, ignore_index = True) #vertically stack 1 dataset on top of the other
combined_data = pd.concat([combined_data,dataset_2], axis = 0, ignore_index = True) #append custom dataset

#shuffle dataframe for even dispersion of items throughout, needed for splitting into test and train datasets
combined_data = combined_data.sample(frac=1)

#preprocess truth and lie data, split into train, test (80% train, 20% test), do this later i just want to see something work
data_length = len(combined_data)
print(data_length)

train_data = combined_data.iloc[0:(int(data_length*0.8))] #~80%
test_data = combined_data.iloc[((int(data_length*0.8))+1):(data_length)] # the rest of the data (~20%)

#set up class architecture so DataLoader works properly in optimization
class MisinformationDetectionTraining(Dataset):
    def __init__(self): 
        self.text = train_data["text"].tolist()
        self.labels = train_data["Category"].tolist()

    def __len__(self):
        return len(self.text)

    def __getitem__(self,idx):
        text = (self.text[idx]) #text at that position
        #print(text)
        text = "".join(text) #adds the text to empty string cleanly with no brackets
        #print(text)
        label = int(self.labels[idx]) #label at that position
#        if random.randint(0,(15*batch_size)) == 100*batch_size: #2 per epoch
#            text = dataset_2["text"].tolist()
#            rand =random.randint(0,77)
#            text = (text[(rand)])
#            label = dataset_2["Category"].tolist()
#            label = (label[(rand)])

        text = toTensor(text)
        #print(text,label)
        #print(label)
        label = torch.tensor(label)
        return text,label

class MisinformationDetectionTesting(Dataset):
    def __init__(self): 
        self.text = test_data["text"].tolist()
        self.labels = test_data["Category"].tolist()

    def __len__(self):
        return len(self.text)

    def __getitem__(self,idx):
        text = (self.text[idx]) #text at that position
        #print(text)
        text = "".join(text) #adds the text to empty string cleanly with no brackets
        #print(text)
        label = int(self.labels[idx]) #label at that position
        text = toTensor(text)
        #print(text,label)
        #print(label)
        label = torch.tensor(label)
        return text,label
    
class MisinformationDetectionRealTesting(Dataset):
    def __init__(self): 
        self.text = real_dataset["text"].tolist()
        self.labels = real_dataset["Category"].tolist()

    def __len__(self):
        return len(self.text)

    def __getitem__(self,idx):
        text = (self.text[idx]) #text at that position
        #print(text)
        text = "".join(text) #adds the text to empty string cleanly with no brackets
        #print(text)
        label = int(self.labels[idx]) #label at that position
        text = toTensor(text)
        #print(text,label)
        #print(label)
        label = torch.tensor(label)
        return text,label

#testing loop
def test_fun(max_samples):
    with torch.no_grad():
        n_correct = 0
        n_iterations = 0
        n_samples = len(test_loader.dataset)

        for text,labels in test_loader:
            text=text.to("cpu")
            labels = labels.to("cpu")

            prediction = model(text)

            _, predicted = torch.max(prediction,1)
            n_correct += (predicted == labels).sum().item()

            #if n_iterations > max_samples:
            #    break
    
        acc = n_correct/n_samples
        print("test",acc)
        return acc
    
def real_test():
    with torch.no_grad():
        n_correct = 0
        n_samples = len(test_loader.dataset)

        for text,labels in test_loader:
            text=text.to("cpu")
            labels = labels.to("cpu")

            prediction = model(text)

            _, predicted = torch.max(prediction,1)
            n_correct += (predicted == labels).sum().item()
        
        acc = n_correct/n_samples
        print("real_test 36",acc)
        return acc


def train_eval(max_samples):
    with torch.no_grad():
        n_correct = 0
        n_iterations = 0
        n_samples = len(train_loader.dataset)

        for text,labels in train_loader:
            text=text.to("cpu")
            labels = labels.to("cpu")
            n_iterations += 1
            prediction = model(text)

            _, predicted = torch.max(prediction,1)
            n_correct += (predicted == labels).sum().item()

            #if n_iterations > max_samples:
                #break
            
        
        acc = n_correct/n_samples
        print("train",acc)
        return acc

train_loader = DataLoader(dataset=MisinformationDetectionTraining(),batch_size = batch_size, shuffle = True)
test_loader = DataLoader(dataset=MisinformationDetectionTesting(),batch_size = batch_size, shuffle = True)
real_loader = DataLoader(dataset=MisinformationDetectionRealTesting(),batch_size = 16, shuffle = False) #small dataset for real world testing rn
print(train_loader)


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

model = NeuralNetwork().to("cpu")#if using colab and cuda gpu, switch to "cuda"

#print("Network structure:")
#print(model)
signal.signal(signal.SIGINT, signal_handler)

loss_fn = nn.CrossEntropyLoss() #default loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer.zero_grad()

model.train()
def training():
    for epoch in range(0,epochs):
        for i,(text,labels) in enumerate(train_loader): #returns text as tensor, no need to convert again
            #print(text,labels)
            text = text.to("cpu")
            labels = labels.to("cpu")

            #optimization loop
            prediction = model(text)
            loss = loss_fn(prediction,labels)

            #backpropagate
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if (i+1)% 100 == 0:
                print(f'Epoch [{epoch}/{epochs}],Step[{i}],Loss: {float(loss.item())}')
                test =test_fun(500)
                train = train_eval(500)
                real = real_test()
            
                #if abs(train - test) > 0.04: #4%
                 #   print("overfitting detected, terminating.")
                 #   return
            #premature termination
            #if loss <= 0.08 and epoch >= 10:   
                #print("prematurely terminated")
                #break

training()

#save model
torch.save(model.state_dict(), 'E:/Evan/Coding/code/AI/model_architecture/model_weights.pth')

#"real world" testing

#lie = input("Tell a lie or provide a piece of misinformation")
#lie = toTensor(lie)
#prediction = model(lie)
#_, predicted = torch.max(prediction,1)
#print(predicted)   
