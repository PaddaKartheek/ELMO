import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import string
import torch.nn.utils.rnn as padfunc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle

class DownstreamModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers, forwardLM, backwardLM, num_lam, flag=True):
        super(DownstreamModel, self).__init__()
        self.flag = flag
        if self.flag:
            self.num_lam = num_lam
            self.lambdas = nn.Parameter(torch.rand(self.num_lam))
        else:
            self.combine_layer = nn.LSTM(input_dim * 3, input_dim, num_layers=num_layers, batch_first=True)
        self.forwardLM = forwardLM
        self.backwardLM = backwardLM
        self.layer = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.output = nn.Linear(hidden_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)
        for param in self.forwardLM.parameters():
            param.requires_grad = False
        for param in self.backwardLM.parameters():
            param.requires_grad = False
            
    def forward(self, xf, xb):
        fout = self.forwardLM(xf)
        bout = self.backwardLM(xb)
        fupout = [0,0,0]
        bupout = [0,0,0]
        fupout[1], bupout[1], fupout[2], bupout[2] = fout[2][:, 1:, :], torch.flip(bout[2][:, 1:, :], dims=[1]), fout[3][:, 1:, :], torch.flip(bout[3][:, 1:, :], dims=[1])
        fupout[0], bupout[0] = fout[1][:, 1:, :], torch.flip(bout[1][:, 1:, :], dims=[1])
        ini_embed = torch.cat((fupout[0], bupout[0]), dim=2)
        hn1, hn2 = torch.cat((fupout[1], bupout[1]), dim=2), torch.cat((fupout[2], bupout[2]), dim=2)
        if self.flag:
            final_embed = (self.lambdas[0] * ini_embed) + (self.lambdas[1] * hn1) + (self.lambdas[2] * hn2)
        else:
            concatenated_tensor = torch.cat((ini_embed, hn1, hn2), dim=2)
            final_embed, _ = self.combine_layer(concatenated_tensor)
        out, _ = self.layer(final_embed)
        out = out.mean(dim = 1)
        out = self.output(out)
        return out
    
class ForwardLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout_prob):
        super(ForwardLM, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout_prob)

        self.embed_layer = nn.Embedding(vocab_size, embedding_dim)
        self.layer1 = nn.LSTM(embedding_dim, hidden_dim, batch_first = True)
        self.layer2 = nn.LSTM(hidden_dim, hidden_dim, batch_first = True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        embed = self.embed_layer(x)
        lstm1, _ = self.layer1(embed)
        lstm1 = self.dropout(lstm1)
        lstm2, _ = self.layer2(lstm1)
        lstm2 = self.dropout(lstm2)
        output = self.fc(lstm2)
        return output, lstm1, lstm2, embed


class BackwardLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout_prob):
        super(BackwardLM, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout_prob)

        self.embed_layer = nn.Embedding(vocab_size, embedding_dim)
        self.layer1 = nn.LSTM(embedding_dim, hidden_dim, batch_first = True)
        self.layer2 = nn.LSTM(hidden_dim, hidden_dim, batch_first = True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        embed = self.embed_layer(x)
        lstm1, _ = self.layer1(embed)
        lstm1 = self.dropout(lstm1)
        lstm2, _ = self.layer2(lstm1)
        lstm2 = self.dropout(lstm2)
        output = self.fc(lstm2)
        return output, lstm1, lstm2, embed
    
class MyDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels
    def __len__(self):
        return len(self.embeddings)
    def __getitem__(self, index):
        return self.embeddings[index], self.labels[index]
    
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

dict_path = 'word2idx.pkl'

with open(dict_path, 'rb') as file:
    word2idx = pickle.load(file)

dataset = np.asarray(pd.read_csv('train.csv'))
test_dataset = np.asarray(pd.read_csv('test.csv'))

translation_table = str.maketrans('', '', string.punctuation)
inputs, outputs = [], []
test_inputs, test_outputs = [], []
for index, sent in dataset:
    sent = sent.replace('-', ' ').replace('\\', ' ')
    sent = sent.translate(translation_table)
    sent = sent.split()
    sent = [word.lower() for word in sent if word.isalpha()]
    sent = ['<sos>'] + sent + ['<eos>']
    sent = [word2idx.get(word, word2idx['<unk>']) for word in sent]
    inputs.append(sent)
    outputs.append(torch.tensor(index - 1, dtype=torch.long))

for index, sent in test_dataset:
    sent = sent.replace('-', ' ').replace('\\', ' ')
    sent = sent.translate(translation_table)
    sent = sent.split()
    sent = [word.lower() for word in sent if word.isalpha()]
    sent = ['<sos>'] + sent + ['<eos>']
    sent = [word2idx.get(word, word2idx['<unk>']) for word in sent]
    test_inputs.append(sent)
    test_outputs.append(torch.tensor(index - 1, dtype=torch.long))

finputs, binputs = [], []
for inp in inputs:
    f = inp[:-1]
    b = inp[1:]
    b = b[::-1]
    finputs.append(torch.tensor(f))
    binputs.append(torch.tensor(b))

pad_finputs, pad_binputs = padfunc.pad_sequence(finputs, batch_first=True), padfunc.pad_sequence(binputs, batch_first=True)

com_inputs = list(zip(pad_finputs, pad_binputs))

fpath = 'forward1.pt'
fmodel = ForwardLM(len(word2idx) + 1, 300, 300, 0.5)
fmodel.load_state_dict(torch.load(fpath, map_location=device))

bpath = 'backward1.pt'
bmodel = BackwardLM(len(word2idx) + 1, 300, 300, 0.5)
bmodel.load_state_dict(torch.load(bpath, map_location=device))

num_epochs = 5
batch_size = 256
learning_rate = 0.001

model = DownstreamModel(600, 300, 4, 1, fmodel, bmodel, 3)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
model = model.to(device)


train_loader = DataLoader(dataset=MyDataset(com_inputs, outputs), batch_size=batch_size, shuffle=True)

# To freeze Lambdas

# for param in model.named_parameters():
#     if param[0] == 'lambdas':
#         param[1].requires_grad = False

model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for batch_input, batch_output in tqdm(train_loader, total = len(train_loader), desc="Training"):
        batch_output = batch_output.to(device)
        batch_input_0 = batch_input[0].to(device)
        batch_input_1 = batch_input[1].to(device)
        optimizer.zero_grad()
        model_output = model(batch_input_0, batch_input_1)
        loss = criterion(model_output, batch_output)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_input_0.size(0)
    epoch_train_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss: {epoch_train_loss:.4f}")
    
model_path = 'classifier_trainable.pt'
torch.save(model.state_dict(), model_path)

test_finputs, test_binputs = [], []
for inp in test_inputs:
    f = inp[:-1]
    b = inp[1:]
    b = b[::-1]
    test_finputs.append(torch.tensor(f))
    test_binputs.append(torch.tensor(b))

test_pad_finputs, test_pad_binputs = padfunc.pad_sequence(test_finputs, batch_first=True), padfunc.pad_sequence(test_binputs, batch_first=True)
test_com_inputs = list(zip(test_pad_finputs, test_pad_binputs))
test_loader = DataLoader(dataset=MyDataset(test_com_inputs, test_outputs), batch_size=batch_size, shuffle=True)

model.eval()
predicted_labels = []
true_labels = []
with torch.no_grad():
    for batch_input, batch_output in tqdm(test_loader, total=len(test_loader), desc="Testing"):
        batch_output = batch_output.to(device)
        batch_input_0 = batch_input[0].to(device)
        batch_input_1 = batch_input[1].to(device)
        model_output = model(batch_input_0, batch_input_1)
        pred = model_output.argmax(dim=1)
        predicted_labels.extend(pred.cpu().numpy())
        true_labels.extend(batch_output.cpu().numpy())

accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='macro')
recall = recall_score(true_labels, predicted_labels, average='macro')
f1 = f1_score(true_labels, predicted_labels, average='macro')
conf_matrix = confusion_matrix(true_labels, predicted_labels)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:\n", conf_matrix)