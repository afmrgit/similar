from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import numpy as np
reference_file = "./reference.txt"
reference_file2 = "./reference2.txt"

with open(reference_file, "r") as f:
  sentence1 = f.read().strip()

with open(reference_file2, "r") as f:
  sentence2 = f.read().strip()


# Define the calculate_similarity function
def calculate_similarity(sentence1, sentence2):
    # Tokenize the sentences
    tokens1 = simple_preprocess(sentence1)
    tokens2 = simple_preprocess(sentence2)

    # Load or train a Word2Vec model
    # Here, we'll create a simple model for demonstration purposes
    sentences = [tokens1, tokens2]
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=0)

    # Calculate the vector representation for each sentence
    vector1 = np.mean([model.wv[token] for token in tokens1], axis=0)
    vector2 = np.mean([model.wv[token] for token in tokens2], axis=0)

    # Calculate cosine similarity
    similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    return similarity

# Example usage
similarity_score = calculate_similarity(sentence1, sentence2)
print("Similarity score:", similarity_score)
