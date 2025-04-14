from openalex import fetch_abstracts
from embedding import Embedder
from faiss_index import FaissIndex
from similarity import compute_agreement
import numpy as np

def verify_claim(claim, search_query):
    print(f"\nClaim: \"{claim}\"\nSearching papers for: \"{search_query}\"\n")

    abstracts = fetch_abstracts(search_query)
    if not abstracts:
        print("No abstracts found.")
        return

    embedder = Embedder()
    paper_embeddings = embedder.embed(abstracts).cpu().detach().numpy()
    claim_embedding = embedder.embed([claim]).cpu().detach().numpy()

    index = FaissIndex(dim=paper_embeddings.shape[1])
    index.add(paper_embeddings, abstracts)

    agreement, similarities = compute_agreement(claim_embedding, paper_embeddings)
    print(f"Agreement Score: {agreement}%\n")

    print("Top Papers (most similar):")
    top_k = min(5, len(abstracts))
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    for i in top_indices:
        print(f"- Similarity: {round(similarities[i]*100, 2)}% | Abstract: {abstracts[i][:150]}...")

if __name__ == "__main__":
    claim = "Red meat consumption increases risk of heart disease."
    search_query = "red meat heart disease"
    verify_claim(claim, search_query)
