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
import xgboost as xgb
from scipy.sparse import hstack,vstack,csc_matrix
from sklearn.model_selection import cross_val_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import text_preprocess
import node2vec


path_to_data = 'data/'
use_saved = True
n_class = 8

n_walks = 380
walk_length = 30
p = 0.25
q = 4
window_size = 5
iter = 3

def visualize(model, n, dim):
    # Plot node embeddings in two-dimension space
    nodes = train_hosts
    vecs = np.empty(shape=(n, dim))
    
    for i in range(n):
        vecs[i, :] = model.wv[nodes[i]]
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k','pink']
    class_mapping = {label:idx for idx,label in enumerate(np.unique(y_train))}
    y_ = pd.Series(y_train).map(class_mapping).values


    my_tsne = TSNE(n_components=2)

    vecs_tsne = my_tsne.fit_transform(vecs)
    print(vecs.shape)
    fig, ax = plt.subplots()
    for i in range(vecs_tsne.shape[0]):
        ax.scatter(vecs_tsne[i,0], vecs_tsne[i,1],label=i, c  = colors[y_[i]],s=20)

    fig.suptitle('t-SNE visualization of node embeddings',fontsize=30)
    fig.set_size_inches(15,9)
    plt.show()


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

train_data = list()
for host in train_hosts:
    if host in text_cleaned:
        train_data.append(' '.join(text_cleaned[host]))
    else:
        train_data.append('')

test_data = list()
for host in test_hosts:
    if host in text_cleaned:
        test_data.append(' '.join(text_cleaned[host]))
    else:
        test_data.append('')


"""
    Train node2vec + TF-IDF
"""
# Create the tf-idf feature matrix
vec = TfidfVectorizer(decode_error='ignore', strip_accents='unicode', encoding='latin-1', min_df=10, max_df=1000)
X_train_text = vec.fit_transform(train_data)
X_test_text = vec.transform(test_data)

print("Train matrix dimensionality: ", X_train_text.shape)
print("Test matrix dimensionality: ", X_test_text.shape)

# Train node2vec
model = Node2Vec(G, walk_length=walk_length, num_walks=n_walks, p=p, q=q, workers=1) #init model
w2v_model = model.train(window_size=window_size, iter=iter) # train model
embeddings = model.get_embeddings()


X_train = np.asarray([embeddings[i] for i in train_hosts])
X_test = np.asarray([embeddings[i] for i in test_hosts])

X_train1 = hstack((X_train, X_train_text))
X_test1 = hstack((X_test, X_test_text))

clf = xgb.XGBClassifier(objective='multi:softmax')
scores_merge = -cross_val_score(clf, X_train1, y_train, cv=5, scoring='neg_log_loss', n_jobs=-1)

print("mean: %e (+/- %e)" % (scores_merge.mean(), scores_merge.std()))


# Prediction on test set
params = {
    'objective': 'multi:softmax',
    'eval_metric': 'mlogloss', 
    'verbosity': 1,
    'eta': 0.0763189143733192, 
    'min_child_weight': 18, 
    'max_depth': 40, 
    'max_leaf_nodes': 31, 
    'gamma': 0.5698159938588805, 
    'max_delta_step': 30, 
    'subsample': 0.7379104404086604, 
    'max_features': 0.567830522637841, 
    'lambda': 1.228598904771292e-08, 
    'alpha': 1.1681312757510824, 
    'scale_pos_weight': 0.3829393944627685
}
clf = xgb.XGBClassifier(**params)
clf.fit(X_train1, y_train)
y_pred = clf.predict_proba(X_test1)

# Write predictions to a file
with open('node2_vec_cleaned.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    lst = clf.classes_.tolist()
    lst.insert(0, "Host")
    writer.writerow(lst)
    for i,test_host in enumerate(test_hosts):
        lst = y_pred[i,:].tolist()
        lst.insert(0, test_host)
        writer.writerow(lst)


visualize(w2v_model, 2000, 128)


"""
    Hyperparameters tuning 
"""
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

cv = StratifiedKFold(n_splits=5)

def objective(trial):
    eta = trial.suggest_loguniform('eta', 1e-3, 1e-0)
    min_child_weight = trial.suggest_int('min_child_weight', 1, 50)
    max_depth = trial.suggest_int('max_depth', 1, 50)
    max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 0, 50)
    gamma = trial.suggest_uniform('gamma', 0, 1.0)
    max_delta_step = trial.suggest_int('max_delta_step', 0, 40)
    subsample = trial.suggest_uniform('subsample', 0.1, 1.0)
    max_features = trial.suggest_uniform('max_features', 0.1, 1.0)
    lambda_ = trial.suggest_loguniform('lambda', 1e-8, 1.0)
    alpha = trial.suggest_loguniform('alpha', 1e-0, 2)
    scale_pos_weight = trial.suggest_loguniform('scale_pos_weight', 0.1, 2)

    params = {
        'silent': 1,
        'objective': 'multi:softmax',
        'eval_metric': 'mlogloss', 
        'eta': eta,
        'min_child_weight': min_child_weight,
        'max_depth': max_depth,
        'max_leaf_nodes': max_leaf_nodes,
        'gamma': gamma,
        'max_delta_step': max_delta_step,
        'subsample': subsample,
        'max_features': max_features,
        'lambda': lambda_,
        'alpha': alpha,
        'scale_pos_weight': scale_pos_weight,
    }
    regressor = xgb.XGBClassifier(**params)

    return np.mean(-cross_val_score(regressor, X_train1, y_train, cv=cv, scoring='neg_log_loss', n_jobs=-1))

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# Show best result
print(study.best_trial.params)
print(study.best_trial.value)
