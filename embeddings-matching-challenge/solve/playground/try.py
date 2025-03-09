#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

# for train data
train_data = pd.read_csv('./files/train_data.csv')
train_embeddings = pd.read_csv('./files/train_embeddings.csv')
train_labels = pd.read_csv('./files/train_labels.csv')
train_embeddings_similarity = cosine_similarity(train_embeddings)
train_data_similarity = cosine_similarity(train_data)

# for test data
test_embeddings = pd.read_csv('./files/test_embeddings.csv')
test_data = pd.read_csv('./files/test_data.csv')
test_embeddings_similarity = cosine_similarity(test_embeddings)
test_data_similarity = cosine_similarity(test_data)

# scipy.sparse matrices for efficient computation of cosine similarities
train_embeddings_similarity_sparse = csr_matrix(train_embeddings_similarity)
train_data_similarity_sparse = csr_matrix(train_data_similarity)

num_images = train_data_similarity.shape[0]
num_embeddings = train_embeddings_similarity.shape[0]

# Upper triangle indices for train data similarity
train_upper_triangle_indices_data = np.triu_indices(num_images, k=1)

# Upper triangle indices for train embeddings similarity
train_upper_triangle_indices_embeddings = np.triu_indices(num_embeddings, k=1)

# Step 2: Extract the upper triangular part of the similarity matrix
# For train data similarity
train_upper_triangle_similarities_data = train_data_similarity_sparse[train_upper_triangle_indices_data]

# For train embeddings similarity (data-to-embedding similarity)
train_upper_triangle_similarities_embeddings = train_embeddings_similarity_sparse[train_upper_triangle_indices_embeddings]

# Link image pairs with embedding pairs based on similarity
threshold = 0.05  # Define threshold for cosine similarity closeness

# Create a list to store the linked pairs
linked_pairs = []

# Find equal similarities (where cosine similarity values are equal)
equal_similarities_indices = np.isclose(train_upper_triangle_similarities_data, train_upper_triangle_similarities_embeddings, atol=threshold)

# test for now...
something = pd.DataFrame(equal_similarities_indices)
print(something)
something.to_csv('./files/linked_pairs.csv', index=False)

# test for now...
something = pd.DataFrame(train_upper_triangle_similarities_data)
print(something)
something.to_csv('./files/linked_pairs.csv', index=False)

# Store the linked pairs in a DataFrame
linked_pairs_df = pd.DataFrame(linked_pairs, columns=["ImagePair", "EmbeddingPair"])

# Print the resulting DataFrame
print(linked_pairs_df)

