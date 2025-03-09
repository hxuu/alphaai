#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# Compute cosine similarity between two datasets
def cosine_similarity_between_matrices(data1, data2):
    values1 = data1.values
    values2 = data2.values

    # Normalize rows of each matrix
    normalized1 = values1 / np.linalg.norm(values1, axis=1, keepdims=True)
    normalized2 = values2 / np.linalg.norm(values2, axis=1, keepdims=True)

    # Compute cosine similarity matrix
    similarity_matrix = np.dot(normalized1, normalized2.T)
    return similarity_matrix


def find_neighbors(cosine_similarity_matrix, top_n=5):
    # Use argpartition to get indices of top_n neighbors
    top_indices = np.argpartition(-cosine_similarity_matrix, range(top_n), axis=1)[:, :top_n]

    # Sort these indices by actual similarity values
    sorted_indices = np.argsort(-cosine_similarity_matrix[np.arange(cosine_similarity_matrix.shape[0])[:, None], top_indices], axis=1)
    top_sorted_indices = top_indices[np.arange(top_indices.shape[0])[:, None], sorted_indices]

    return {i: top_sorted_indices[i] for i in range(cosine_similarity_matrix.shape[0])}


def link_images_to_embeddings(image_neighbors, embedding_neighbors, train_labels, top_n=5):
    image_col = train_labels.columns[0]
    label_col = train_labels.columns[1]

    train_labels_set = set(zip(train_labels[image_col], train_labels[label_col]))
    result_labels = []

    # Fill all rows by matching each image to the best embedding
    for image_index, img_neighbors in image_neighbors.items():
        best_embedding = None
        max_overlap = 0

        for embedding_index, emb_neighbors in embedding_neighbors.items():
            # Find overlap between image neighbors and embedding neighbors
            overlap = len(set(img_neighbors) & set(emb_neighbors))

            # Prioritize matches in train_labels_set
            if overlap > max_overlap or (overlap == max_overlap and (embedding_index, image_index) in train_labels_set):
                max_overlap = overlap
                best_embedding = embedding_index

        # Add the best embedding for the current image
        if best_embedding is not None:
            result_labels.append((image_index, best_embedding))

    return result_labels


# Load the data
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

# Compute the cosine similarity between test_data and train_data
image_similarity = cosine_similarity_between_matrices(test_data, train_data)

# Compute the cosine similarity between test_embeddings and train_embeddings
embedding_similarity = cosine_similarity_between_matrices(test_embeddings, train_embeddings)

# Find neighbors for images and embeddings
image_neighbors = find_neighbors(image_similarity, top_n=5)
embedding_neighbors = find_neighbors(embedding_similarity, top_n=5)

# Generate result labels by linking images to embeddings
result_labels = link_images_to_embeddings(image_neighbors, embedding_neighbors, train_labels)

# Save results to a CSV file
result_df = pd.DataFrame(result_labels, columns=['row ID', 'label'])
result_df_sorted = result_df.sort_values(by='row ID', ascending=True)

if len(result_df_sorted) < len(test_data):
    print(f"Warning: Only {len(result_df_sorted)} rows were generated. Filling remaining rows with default mappings.")

# Fill missing rows if necessary
all_image_indices = set(range(len(test_data)))
current_image_indices = set(result_df_sorted['row ID'])
missing_image_indices = all_image_indices - current_image_indices

for missing_image_index in missing_image_indices:
    result_df_sorted = pd.concat([result_df_sorted, pd.DataFrame([[missing_image_index, -1]], columns=['row ID', 'label'])])

result_df_sorted = result_df_sorted.sort_values(by='row ID', ascending=True)

result_df_sorted.to_csv('./files/result_labels.csv', index=False)
print("Results have been saved to 'result_labels.csv'.")

