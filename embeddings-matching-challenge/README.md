# Embeddings Matching Challenge

Documentation and writeup for the challenge.
Link every image with its coressponding embeddings.

## Analysis (Introduction)

I'm presented with the following data that I yet again I can't understand:

```bash
❯ tree
.
├── test_data.csv
├── test_embeddings.csv
├── train_data.csv
├── train_embeddings.csv
└── train_labels.csv

1 directory, 5 files
```

#### Explanation of the challenge

This challenge involves matching embeddings to their corresponding images.

An embedding is a numerical representation of data (e.g., an image) in a lower-dimensional
space, where similar data points (e.g., similar images) are closer in this space.

- The key tasks of this challenge is:

1. Understand the relationship between the image which is in an upper dimension
and the embeddings which are in lower dimensions that the original.

2. Match the embeddings in the test set to their corresponding images in the
image set, to generate a csv file just like the test labels.

=> The challenge relies heavily the concept of `cosine similarity`, which we'll
understand now.

#### Understanding cose similarity

The cosine similarity is an easy metric to calculate how 'similar' or 'different'
things are, for example:

```
I love chess. [positive]
I love chess! [positive]
I like playing chess so much!!! [positive]
I hate* chess [negative]
```

The approach is very simple as highlighted by [this video](https://www.youtube.com/watch?v=e9U0QAFbfLI):

1. Make a word table with each word having a count in its respective sentence
2. Then, plot a graph with the word and their count.
3. Plot each sentence.
4. the cosine of the angle between every dot (that is a sentence) represents the `cosine similarity`
between those sentences. `[0-1]` (0 being no similarity and 1 being same)

The above steps are complicated to calculate for EVERY sentence and word. that's why
we have an already ready formula. (The one given in the challenge)

#### Understanding the data

`train_data.csv` : a subset of 128 of the original training images.
`train_embedding.csv` : the corresponding embeddingd to the images in test_data.csv.
`train_labels.csv` : contains the correspondence between the training images and the training embeddings,for example if :

```
train_labels = [8703,2000,1181,...]
```
-> it means that the first embedding in the dataset corresponds to the $ 8703^{th} $ image.

Similarly `test_data.csv` and `test_embeddings.csv` are provided.

#### Gaps that need to be filled

- loss function

- hint: the distance between two images, is relatively the same distance between
their corresponding embeddings.

## Building the Solution

### Understanding the problem

We have to connect the embeddings to their corresponding images. How to do that do, idk?

The problem is that cosine similarity is calculated using loops, our job is to calculate
cosine similarities using matrix multiplication only

check [this](https://stats.stackexchange.com/questions/160080/calculating-cosine-similarity-with-matrix-decomposition-matrix-multiplication-w/161267#161267)
