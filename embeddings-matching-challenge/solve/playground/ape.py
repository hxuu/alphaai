#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load data
train_data = pd.read_csv('./files/train_data.csv')
train_embeddings = pd.read_csv('../challenge/train_embeddings.csv')
train_labels = pd.read_csv('./files/train_labels.csv')

data1 = train_data.loc[87].values.reshape(1, -1)
data2 = train_data.loc[105].values.reshape(1, -1)
data3 = train_data.loc[64].values.reshape(1, -1)
data4 = train_data.loc[127].values.reshape(1, -1)

embd1 = train_embeddings.loc[0].values.reshape(1, -1)
embd2 = train_embeddings.loc[1].values.reshape(1, -1)
embd3 = train_embeddings.loc[2].values.reshape(1, -1)
embd4 = train_embeddings.loc[3].values.reshape(1, -1)

similarity1 = cosine_similarity(embd1, embd2)
similarity2 = cosine_similarity(data1, data2)
similarity3 = cosine_similarity(embd3, embd4)
similarity4 = cosine_similarity(data3, data4)

# print(f"Cosine similarity1: {similarity1[0][0]}")
# print(f"Cosine similarity2: {similarity2[0][0]}")
print(f"Cosine similarity: {similarity3[0][0]}")
print(f"Cosine similarity: {similarity4[0][0]}")

