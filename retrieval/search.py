import json
import faiss
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

load_dotenv()

client = OpenAI()

INDEX_PATH = "data/index.faiss"
METADATA_PATH = "data/metadata.json"

def load_index():
    return faiss.read_index(INDEX_PATH)

def load_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def get_query_embeddings(query):
    response = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )

    embedding = np.array(
        response.data[0].embedding,
        dtype=np.float32
    )
    return embedding.reshape(1,-1)

def search(index, metadata, question, k=5):
    query_embedding = get_query_embeddings(question)
    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(
        query_embedding,
        k
    )

    result = []

    for distance, idx in zip(distances[0], indices[0]):

        if idx == -1:
            continue
        if distance < 0.40:
            continue
        chunk = metadata[idx]

        result.append({
            "score": distance,
            "chunk": chunk
        })

    return result

def display_result(results):
    print("\n Search Results\n")

    for i, result in enumerate(results, start=1):
        chunk = result["chunk"]

        print("="*80)

        print(f"Result {i}")
        print(f"Distance {result['score']:.2f}")
        print(f"Lecture: {chunk['title']}")
        print(f"Time -> {chunk['start']:.2f} - {chunk['end']:.2f}")

        print()
        print(chunk["text"])
        print()

def main():
    index = load_index()
    metadata = load_metadata()

    while True:
        question = input("\nAsk a question (or type 'exit'): ")

        if question.lower() == "exit":
            break

        results = search(index, metadata, question)

        display_result(results)

if __name__ == "__main__":
    main()

