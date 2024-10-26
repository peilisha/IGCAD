import os.path
import pickle
import numpy as np
import torch
import networkx as nx
from tqdm import tqdm
from sklearn.metrics import mutual_info_score
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0  #
    similarity = dot_product / (norm1 * norm2)
    return similarity
def create_edge_index2(features, threshold=0.9):  #
    # features = features.transpose(0, 2, 1)
    num_graphs = features.shape[0]
    num_nodes = features.shape[1]
    adjacency_matrix = torch.zeros(num_graphs, num_nodes, num_nodes)
    edge_indexs = [None] * num_graphs
    for k in tqdm(range(num_graphs)):
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                similarity = cosine_similarity(features[k][i], features[k][j])
                # print("similarity:",similarity)#
                if similarity >= threshold:
                    adjacency_matrix[k][i][j] = 1
                    adjacency_matrix[k][j][i] = 1
    for k in tqdm(range(num_graphs)):
        adjacency_matrixk = sp.coo_matrix(adjacency_matrix[k])
        values = adjacency_matrixk.data
        indices = np.vstack((adjacency_matrixk.row, adjacency_matrixk.col))
        edge_indexs[k] = torch.LongTensor(indices)

    return edge_indexs
