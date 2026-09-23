import os 
import json
from openai import OpenAI
from dotenv import load_dotenv
import hashlib

load_dotenv()
client = OpenAI()

CHUNK_DIR = "data/chunks"
EMBED_DIR = "data/embeddings"

os.makedirs(EMBED_DIR, exist_ok=True)

def compute_hash(chunks):
    texts = [chunk["text"] for chunk in chunks]
    combine = "\n---\n".join(texts)
    return hashlib.md5(combine.encode()).hexdigest()

def load_chunks(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_embeddings(text):
    response = client.embeddings.create(input=text, model="text-embedding-3-small")
    return response.data

def embed_chunks(chunks, filename):
    text = [chunk["text"] for chunk in chunks]

    print(f"Embedding file {filename}")
    embedded_chunks = get_embeddings(text)

    result = []

    for chunk, embedding in zip(chunks, embedded_chunks):
        result.append({
            **chunk,
            "embedding": embedding.embedding
        })
    return result

def save_embed(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )

def main():
    for filename in os.listdir(CHUNK_DIR):

        if not filename.endswith(".json"):
            continue

        chunk_path = os.path.join(CHUNK_DIR,filename)

        output_path = os.path.join(EMBED_DIR, filename)


        chunks = load_chunks(chunk_path)
        current_hash = compute_hash(chunks)

        if os.path.exists(output_path):
            existing = load_chunks(output_path)
            if isinstance(existing, dict) and existing.get("source_hash") == current_hash:   
                print(f"Skipping {filename} (unchanged)")
                continue
        embedded_chunks = embed_chunks(chunks, filename)

        save_embed({"source_hash": current_hash, "chunks": embedded_chunks},
                   output_path
                )

        print(f"Saved {len(embedded_chunks)} chunks")

if __name__ == "__main__":
    main()

