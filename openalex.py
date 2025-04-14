import requests

def fetch_abstracts(query, num_papers=10):
    url = "https://api.openalex.org/works"
    params = {
        "filter": f"title.search:{query}",
        "per-page": num_papers
    }
    response = requests.get(url, params=params)
    data = response.json()
    abstracts = []

    for item in data.get("results", []):
        abstract = item.get("abstract_inverted_index")
        if abstract:
            # Convert OpenAlex inverted index back to string
            word_list = [""] * max([max(v) for v in abstract.values()])
            for word, positions in abstract.items():
                for pos in positions:
                    word_list[pos] = word
            abstract_text = " ".join(word_list)
            abstracts.append(abstract_text)

    return abstracts
