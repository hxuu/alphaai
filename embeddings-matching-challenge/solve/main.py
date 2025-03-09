#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def cosine_similarity(data):
    R = data.values
    Rbar = R / np.linalg.norm(R, axis=1, keepdims=True)
    return np.dot(Rbar, Rbar.T)

def find_neighbors(cosine_similarity_matrix, top_n=5):
    # Use argpartition to get indices of top_n+1 neighbors (including self)
    top_indices = np.argpartition(-cosine_similarity_matrix, range(1, top_n + 1), axis=1)[:, 1:top_n+1]

    # Sort these top indices based on the actual similarity values
    sorted_indices = np.argsort(-cosine_similarity_matrix[np.arange(cosine_similarity_matrix.shape[0])[:, None], top_indices], axis=1)
    top_sorted_indices = top_indices[np.arange(top_indices.shape[0])[:, None], sorted_indices]

    # Return as a dictionary
    return {i: top_sorted_indices[i] for i in range(cosine_similarity_matrix.shape[0])}


def link_images_to_embeddings(image_neighbors, embedding_neighbors, train_labels, top_n=5):
    image_col = train_labels.columns[0]
    label_col = train_labels.columns[1]

    train_labels_set = set(zip(train_labels[image_col], train_labels[label_col]))
    result_labels = set()

    image_neighbors_sets = {i: set(neighbors) for i, neighbors in image_neighbors.items()}
    embedding_neighbors_sets = {i: set(neighbors) for i, neighbors in embedding_neighbors.items()}

    for image_index, img_neighbors_set in image_neighbors_sets.items():
        for embedding_index, emb_neighbors_set in embedding_neighbors_sets.items():
            # Find common neighbors directly
            common_neighbors = img_neighbors_set & emb_neighbors_set

            # Filter the pairs present in train_labels_set
            valid_pairs = {(emb, img) for img in common_neighbors for emb in common_neighbors if (emb, img) in train_labels_set}

            if len(valid_pairs) >= 2:
                result_labels.add((image_index, embedding_index))
                break  # Exit inner loop as the condition is satisfied

    return result_labels

train_labels = pd.read_csv('./files/train_labels.csv')
train_data = pd.read_csv('./files/train_data.csv')
train_embeddings = pd.read_csv('./files/train_embeddings.csv')
test_data = pd.read_csv('./files/test_data.csv')
test_embeddings = pd.read_csv('./files/test_embeddings.csv')

print(train_labels.shape)
print(test_data.shape)
print(test_embeddings.shape)
print(train_data.shape)
print(train_embeddings.shape)

image_similarity = cosine_similarity(test_data)
embedding_similarity = cosine_similarity(test_embeddings)

image_neighbors = find_neighbors(image_similarity, top_n=15)
embedding_neighbors = find_neighbors(embedding_similarity, top_n=15)

result_labels = link_images_to_embeddings(image_neighbors, embedding_neighbors, train_labels)

result_df = pd.DataFrame(result_labels, columns=['row ID', 'label'])
result_df_sorted = result_df.sort_values(by='row ID', ascending=True)

if len(result_df_sorted) < 10000:
    print("Warning: Only", len(result_df_sorted), "rows were generated. Consider adjusting parameters.")

result_df_sorted.to_csv('./files/result_labels.csv', index=False)
print("Results have been saved to 'result_labels.csv'.")
