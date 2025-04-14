# AI-Based Scientific Fact Checker

This project is an AI-driven tool designed to verify the validity of scientific claims by comparing them against research papers and abstracts. It uses a combination of machine learning and natural language processing (NLP) techniques to assess whether a claim is supported by the current body of scientific knowledge.

## Project Overview

The system uses a SciBERT-based Retrieval-Augmented Generation (RAG) pipeline to evaluate how well a claim aligns with scientific data. The verification process involves comparing the claim with scientific literature and outputting an agreement percentage based on cosine similarity between the claim and relevant research papers.

### Key Features

- **SciBERT-based Search**: Leverages SciBERT, a transformer model, to embed claims and research abstracts for comparison.
- **FAISS for Fast Search**: Implements FAISS (Facebook AI Similarity Search) for fast and efficient similarity search.
- **Cosine Similarity**: Measures the cosine similarity between the claim embedding and abstract embeddings.
- **OpenAlex Integration**: Fetches research paper abstracts from the OpenAlex database to compare with the user's claim.

## Usage Example

### Claim Verification Example

Below is an example of how you can verify a scientific claim using this tool:

```python
claim = "Red meat consumption increases risk of heart disease."
search_query = "red meat heart disease"
verify_claim(claim, search_query)
Claim: 'Red meat consumption increases risk of heart disease.'
Searching papers for: 'red meat heart disease'

Agreement: 75% (3 out of 4 papers agree with the claim)
Papers used:
- Abstract of Paper 1: "Study showing the link between red meat and heart disease..."
- Abstract of Paper 2: "Analysis of cardiovascular risks associated with red meat consumption..."
- Abstract of Paper 3: "Impact of red meat on heart health: A review..."
