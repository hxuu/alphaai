#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression  # Example model for embedding prediction

# Load the training data (image-embedding pairs)
train_labels = pd.read_csv('./files/train_labels.csv')

# Load the test data and test embeddings
test_data = pd.read_csv('./files/test_data.csv')
test_embeddings = pd.read_csv('./files/test_embeddings.csv')

# Extract features and embeddings from the training data
train_embeddings = train_labels.iloc[:, :-1].values.reshape(1, -1)
train_images = train_labels.iloc[:, -1].values.reshape(1, -1)

model = LinearRegression()
model.fit(train_images, train_embeddings)  # Train the model

# Function to predict the embedding for a test image
def image_to_embedding(image_data):
    """
    Predict the embedding for a given image.
    """
    return model.predict(image_data)  # Use the trained model to predict the embedding

# Initialize a list to store the corresponding labels
test_labels = []

# Loop through each test embedding and predict its label
for i, test_row in test_embeddings.iterrows():
    test_embedding = test_row.values.reshape(1, -1)  # Reshape if necessary

    predicted_embedding = image_to_embedding(test_embedding)
    similarities = cosine_similarity(predicted_embedding, test_embeddings)

    best_match_index = np.argmax(similarities)
    test_labels.append(best_match_index)

labels_df = pd.DataFrame({'Unnamed': range(len(test_labels)), 'Labels': test_labels})
labels_df.to_csv('./files/test_labels.csv', index=False)

print(f"Labels have been generated and saved to './files/test_labels.csv'.")

