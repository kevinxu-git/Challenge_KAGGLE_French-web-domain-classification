import numpy as np
import networkx as nx
from random import randint
from gensim.models import Word2Vec

def random_walk(G, node, walk_length):
    walk = [node]
    for i in range(walk_length):
        neighbors = list(G.neighbors(walk[-1]))
        if len(neighbors) > 0:
            walk.append(neighbors[randint(0, len(neighbors)-1)])
        else:
            break
    walk = [str(node) for node in walk]
    return walk


def generate_walks(G, num_walks, walk_length):
    walks = []
    for i in range(num_walks):
        for node in G.nodes():
            walks.append(random_walk(G, node, walk_length))
    return walks

def deepwalk(G, num_walks, walk_length, n_dim):
    print("Generating walks")
    walks = generate_walks(G, num_walks, walk_length)

    print("Training word2vec")
    model = Word2Vec(size=n_dim, window=8, min_count=0, sg=1, workers=8)
    model.build_vocab(walks)
    model.train(walks, total_examples=model.corpus_count, epochs=5)
    return model