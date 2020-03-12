import os
import re 
import csv
import codecs
from os import path
import pickle
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import accuracy_score, log_loss
from sklearn.feature_extraction.text import TfidfVectorizer

import text_preprocess
import GNN


path_to_data = 'data/'
use_saved = True
n_class = 8


"""
	Load data
"""
# Read training data
with open(path_to_data + 'train.csv', 'r') as f:
    train_data = f.read().splitlines()

train_hosts = list()
y_train = list()
for row in train_data:
    host, label = row.split(',')
    train_hosts.append(host)
    y_train.append(label.lower())

# Read test data
with open(path_to_data + 'test.csv', 'r') as f:
    test_hosts = f.read().splitlines()

# Create a directed, weighted graph
G = nx.read_weighted_edgelist(path_to_data + 'edgelist.txt', create_using=nx.DiGraph())
n = G.number_of_nodes()

# Load the textual content of a set of webpages for each host into the dictionary 'text'. 
# The encoding parameter is required since the majority of our text is french.
text = dict()
filenames = os.listdir(path_to_data + 'text/text')
for filename in filenames:
    with codecs.open(path.join(path_to_data + 'text/text', filename), encoding='latin-1') as f: 
        t = f.read().replace('\n', ' ')
        text[filename] = re.sub('\s+', ' ', t)

print("Data loaded !")


"""	
	Text preprocessing 
"""
if use_saved == False:
	text_cleaned = dict()
	for counter, (filename, t) in enumerate(text.items()):
	    # print(counter)
	    my_tokens = clean_text_website_simple(t, my_stopwords=stpwds, punct=punct, remove_stopwords=True, stemming=True)
	    text_cleaned[filename] = my_tokens
	    
	    if counter % 100 == 0:
	        print(counter, 'text processed')

	with open('text_cleaned.p', 'wb') as fp:
	    pickle.dump(text_cleaned, fp, protocol=pickle.HIGHEST_PROTOCOL)
else:
	text_cleaned = pickle.load(open("text_cleaned.p", "rb"))
print("Text cleaned !")


"""
	Train model with Graph Neural Networks
"""
# Define the labels
y_df = pd.DataFrame({'target': y_train})
y_train_facto, label_names = pd.factorize(y_df['target'])

labels_notencoded = list()
train_index = []
for counter, node in enumerate(list(G.nodes())): 
    if node in train_hosts:
        labels_notencoded.append(y_train_facto[train_hosts.index(node)])
        train_index.append(counter)
    else:
        labels_notencoded.append(-1)
labels = labels_notencoded

# Split train/validation set from original train set
idx_train, idx_val = train_test_split(train_index, test_size=0.2, shuffle=True, random_state=42)
idx_test = list()
G_nodes = list(G.nodes())
for node in test_hosts:
    idx_test.append(G_nodes.index(node))

# Define and normalize the adjacency matrix
if use_saved == False:
    adj = nx.to_numpy_matrix(G) # Obtains the adjacency matrix
    adj = normalize_adjacency(adj) # Normalizes the adjacency matrix
    with open('normalized_adj_matrix_sparse.p', 'wb') as fp:
        pickle.dump(adj, fp, protocol=pickle.HIGHEST_PROTOCOL)
else:
    adj = pickle.load(open("normalized_adj_matrix_sparse.p", "rb"))

# Define the node feature matrix
if use_saved == False:
    all_text_data = list()
    for node in list(G.nodes()): 
        all_text_data.append(' '.join(text_cleaned[node]))
    vec = TfidfVectorizer(decode_error='ignore', strip_accents='unicode', encoding='latin-1', min_df=10, max_df=1000)
    features_tfidfVect = vec.fit_transform(all_text_data)
    with open('features_tfidfVect.p', 'wb') as fp:
        pickle.dump(features_tfidfVect, fp, protocol=pickle.HIGHEST_PROTOCOL)
else: 
    features_tfidfVect = pickle.load(open("features_tfidfVect.p", "rb"))
features = features_tfidfVect
print("Node feature matrix :", features.shape)


# Training GNN
device = 'cuda' if torch.cuda.is_available() else 'cpu'
## Hyperparameters
epochs = 20
n_hidden_1 = 128
n_hidden_2 = 64
learning_rate = 0.01
dropout_rate = 0.4
weight_decay = 0.01

## Transforms the numpy matrices/vectors to torch tensors
features = sparse_mx_to_torch_sparse_tensor(features).to(device)
labels = torch.LongTensor(labels).to(device)
adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)
idx_train = torch.LongTensor(idx_train).to(device)
idx_val = torch.LongTensor(idx_val).to(device)

## Creates the model and specifies the optimizer
model = GNN(features.shape[1], n_hidden_1, n_hidden_2, n_class, dropout_rate).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output, _ = model(features, adj) 
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    model.eval()
    output, embeddings = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:03d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    return embeddings[idx_train], embeddings[idx_val]


## Train model
t_total = time.time()
for epoch in range(epochs):
    train(epoch)
    print(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

## Plot embeddings
embeddings_train, embeddings_val = train(100)

plot_tsne(embeddings_train.detach().cpu().numpy(), idx_train.cpu(), labels.detach().cpu().numpy(), label_names, title='T-SNE Visualization of the nodes of the train set')
plot_tsne(embeddings_val.detach().cpu().numpy(), idx_val.cpu(), labels.detach().cpu().numpy(), label_names, title='T-SNE Visualization of the nodes of the validation set')
