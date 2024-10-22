import torch
import torch.nn as nn
import pandas as pd
import string
import pickle
import torch.nn.utils.rnn as padfunc
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

df = pd.read_csv("train.csv")
corpus = df["Description"]

class MyDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels
    def __len__(self):
        return len(self.embeddings)
    def __getitem__(self, index):
        return self.embeddings[index], self.labels[index]

processed_sent = []
unq_words = {}
translation_table = str.maketrans('', '', string.punctuation)
for sent in corpus:
    sent = sent.replace('-', ' ').replace('\\', ' ')
    sent = sent.translate(translation_table)
    sent = sent.split()
    sent = [word.lower() for word in sent if word.isalpha()]
    sent = ['<sos>'] + sent + ['<eos>']
    processed_sent.append(sent)
    for word in sent:
        unq_words[word] = unq_words.get(word, 0) + 1

vocab = []
for key, value in unq_words.items():
    if value >= 10:
        vocab.append(key)

vocab.extend(['<unk>'])
vocab = list(set(vocab))
word2idx = {word: idx + 1 for idx, word in enumerate(vocab)}

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
    
inputs, outputs = [], []

for sent in processed_sent:
    sent = [word2idx.get(word, word2idx['<unk>']) for word in sent]
    inputs.append(torch.tensor(sent[:-1]))
    outputs.append(torch.tensor(sent[1:]))

# Forward LM training

pad_inputs, pad_outputs = padfunc.pad_sequence(inputs, batch_first=True), padfunc.pad_sequence(outputs, batch_first=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

num_epochs = 10
learning_rate = 0.001
batch_size = 256

model = ForwardLM(len(vocab) + 1, 300, 300, 0.5)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
model = model.to(device)

train_loader = DataLoader(dataset=MyDataset(pad_inputs, pad_outputs), batch_size=batch_size, shuffle=True)

model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for batch_input, batch_output in tqdm(train_loader, total = len(train_loader), desc="Training"):
        batch_input, batch_output = batch_input.to(device), batch_output.to(device)
        optimizer.zero_grad()
        output = model(batch_input)
        output_flat = output[0].view(-1, output[0].size(-1))
        target_flat = batch_output.view(-1)
        loss = criterion(output_flat, target_flat)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_input.size(0)
    epoch_train_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss: {epoch_train_loss:.4f}")


model_path = 'forward1.pt'
torch.save(model.state_dict(), model_path)

# Backward LM training

rev_outputs = [torch.flip(inp, [0]) for inp in inputs]
rev_inputs = [torch.flip(oup, [0]) for oup in outputs]

b_pad_inputs, b_pad_outputs = padfunc.pad_sequence(rev_inputs, batch_first=True), padfunc.pad_sequence(rev_outputs, batch_first=True)

num_epochs = 10
learning_rate = 0.001
batch_size = 256

model = BackwardLM(len(vocab) + 1, 300, 300, 0.5)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
model = model.to(device)

train_loader = DataLoader(dataset=MyDataset(b_pad_inputs, b_pad_outputs), batch_size=batch_size, shuffle=True)

model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for batch_input, batch_output in tqdm(train_loader, total = len(train_loader), desc="Training"):
        batch_input, batch_output = batch_input.to(device), batch_output.to(device)
        optimizer.zero_grad()
        output = model(batch_input)
        output_flat = output[0].view(-1, output[0].size(-1))
        target_flat = batch_output.view(-1)
        loss = criterion(output_flat, target_flat)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_input.size(0)
    epoch_train_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss: {epoch_train_loss:.4f}")


model_path = 'backward1.pt'
torch.save(model.state_dict(), model_path)

dict_path = "word2idx.pkl"
with open(dict_path, 'wb') as file:
    pickle.dump(word2idx, file)