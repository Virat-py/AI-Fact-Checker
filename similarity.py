from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def compute_agreement(claim_embedding, paper_embeddings):
    similarities = cosine_similarity(claim_embedding, paper_embeddings)[0]
    avg_similarity = np.mean(similarities)
    return round(avg_similarity * 100, 2), similarities
