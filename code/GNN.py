import scipy.sparse as sp
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.manifold import TSNE


def normalize_adjacency(A):
    # Using sparse matrix
    M = sp.csr_matrix(A)
    n = M.shape[0]
    A_new = M + sp.eye(n)
    D = sp.csr_matrix(np.diag(1/np.sqrt(np.squeeze(np.asarray(sp.csr_matrix.sum(A_new, axis=1).flatten())))))
    A_normalized = D.dot(A_new.dot(D))
    return A_normalized


def accuracy(output, labels):
    """Computes classification accuracy"""
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def plot_tsne(embeddings, idx, labels, label_names, title):
    # Plot node embeddings in two-dimension space
    tsne = TSNE(n_components=2)
    embeddings = tsne.fit_transform(embeddings)

    labels = labels[idx]
    unique_labels = np.unique(labels)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'pink']

    fig, ax = plt.subplots()
    for i in range(unique_labels.size):
        idxs = [j for j in range(labels.size) if labels[j]==unique_labels[i]]
        ax.scatter(embeddings[idxs,0], 
                  embeddings[idxs,1], 
                  c=colors[i],
                  label=label_names[i],
                  alpha=0.7,
                  s=10)

    ax.legend(scatterpoints=1, fontsize=12)
    fig.suptitle(title, fontsize=12)
    fig.set_size_inches(15,9)
    plt.show()


class GNN(nn.Module):
    """Simple GNN model"""
    def __init__(self, n_feat, n_hidden_1, n_hidden_2, n_class, dropout):
        super(GNN, self).__init__()

        self.fc1 = nn.Linear(n_feat, n_hidden_1)
        self.fc1_bn = nn.BatchNorm1d(n_hidden_1)

        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.fc2_bn = nn.BatchNorm1d(n_hidden_2)

        self.fc3 = nn.Linear(n_hidden_2, n_class)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x_in, adj):
        x = self.fc1(x_in)
        x = self.relu(self.fc1_bn(torch.mm(adj, x)))
        x = self.dropout(x)
        
        x = self.fc2(x)
        t = self.relu(self.fc2_bn(torch.mm(adj, x)))

        x = self.fc3(t)
        return F.log_softmax(x, dim=1), t