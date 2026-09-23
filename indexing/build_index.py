import os
import json
import faiss
import numpy as np

EMBEDDING_DIR = "data/embeddings"
INDEX_PATH = "data/index.faiss"
METADATA_PATH = "data/metadata.json"

def load_embeddings():

    all_chunks = []

    for filename in os.listdir(EMBEDDING_DIR):
        if not filename.endswith(".json"):
            continue

        path = os.path.join(EMBEDDING_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
            try:
                chunks = chunks["chunks"]

            except KeyError as e:
                print(f"Failed to load {filename}")
                continue

        for chunk in chunks:
            chunk["source_file"] = filename

        all_chunks.extend(chunks)

    return all_chunks

def build_index(chunks):

    vectors = np.array(
        [
        chunk["embedding"] for chunk in chunks
        ],
        dtype=np.float32
    )

    dimension = vectors.shape[1]

    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(dimension)

    index.add(vectors)

    print(f"Indexed {index.ntotal} vectors")

    return index

def save_metadata(chunks):
    metadata = []

    for chunk in chunks:
        metadata.append({
            "source_file": chunk["source_file"],
            "id": chunk["id"],
            "title": chunk["title"],
            "start": chunk["start"],
            "end": chunk["end"],
            "text": chunk["text"]
        })

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False
        )

def main():
    chunks = load_embeddings()
    index = build_index(chunks)

    faiss.write_index(index,INDEX_PATH)

    save_metadata(chunks)

    print("\nDone!")
    print(f"Index saved to {INDEX_PATH}")
    print(f"Metadata saved to {METADATA_PATH}")

if __name__ == "__main__":
    main()