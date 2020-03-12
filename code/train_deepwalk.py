import csv
import pickle
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

import DeepWalk


path_to_data = 'data/'
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


"""
	DeepWalk
"""
n_dim = 32
n_walks = 30
walk_length = 50

# Generate deepwalk
model = DeepWalk.deepwalk(G, n_walks, walk_length, n_dim)
embeddings = np.zeros((n, n_dim))
for i, node in enumerate(G.nodes()):
    embeddings[i,:] = model.wv[str(node)]
    
X_train = np.zeros((len(train_hosts), n_dim))
for i in range(len(train_hosts)):
    X_train[i,:] = embeddings[int(train_hosts[i]),:]

X_test = np.zeros((len(test_hosts), n_dim))
for i in range(len(test_hosts)):
    X_test[i,:] = embeddings[int(test_hosts[i]),:]


# Classification
clf = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=2000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_train)


scores_merge = -cross_val_score(clf, X_train, y_train, cv=5, scoring='neg_log_loss', n_jobs=-1)

print("mean: %e (+/- %e)" % (scores_merge.mean(), scores_merge.std()))