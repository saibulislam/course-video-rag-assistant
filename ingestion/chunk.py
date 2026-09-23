import os 
import json

TRANSCRIPT_DIR = "data/transcripts"
CHUNK_DIR = "data/chunks"

MAX_CHAR = 500

os.makedirs(CHUNK_DIR, exist_ok=True)

def load_transcripts(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def create_chunks(data):

    chunks = []

    chunk_id = 1
    chunk_start = None
    chunk_end = None
    current_text = ""
    title = data["chunks"][0]["title"]

    for segment in data["chunks"]:

        if chunk_start is None:
            chunk_start = segment["start"]
        chunk_end = segment["end"]
        current_text += " " + segment["text"]

        if len(current_text) >= MAX_CHAR:
            chunks.append({
                "id": chunk_id,
                "title": title,
                "start": chunk_start,
                "end": chunk_end,
                "text": current_text.strip()
            })
            chunk_id += 1
            chunk_start = None
            current_text = current_text[-int(MAX_CHAR*0.10):]

    if current_text:
        chunks.append({
            "id": chunk_id,
            "title": title,
            "start": chunk_start,
            "end": chunk_end,
            "text": current_text.strip()
        })

    return chunks

def save_chunks(chunks, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            chunks,
            f,
            indent=2,
            ensure_ascii=False
        )

def main():
    for filename in os.listdir(TRANSCRIPT_DIR):

        if not filename.endswith(".json"):
            continue

        transcript_path = os.path.join(TRANSCRIPT_DIR, filename)

        data = load_transcripts(transcript_path)

        chunks = create_chunks(data)

        output_path = os.path.join(CHUNK_DIR, filename)

        save_chunks(chunks, output_path)

        print(f"{filename} -> {len(chunks)} chunks")

if __name__ == "__main__":
    main()