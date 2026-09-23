import json
from dotenv import load_dotenv
import faiss
from openai import OpenAI
import numpy as np

load_dotenv()
client = OpenAI()

INDEX_FILE = "data/index.faiss"
METADATA_FILE = "data/metadata.json"

index = faiss.read_index(INDEX_FILE)

with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

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

def search(question, k=5):
    query_embedding = get_query_embeddings(question)
    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(query_embedding, k)

    results = []

    for distance, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        if distance < 0.40:
            continue

        results.append(metadata[idx])

    return results

def build_context(chunks):

    context = ""

    for chunk in chunks:

        context += (
            f"Lecture: {chunk['title']}\n"
            f"File: {chunk['source_file']}\n"
            f"Time: {chunk['start']} - {chunk['end']}\n\n"
            f"{chunk['text']}\n\n"
            f"{'-'*60}\n\n"
        )

    return context

def answer_question(question):
    retrived_chunks = search(question)

    if not retrived_chunks:
        return (
            "I couldn't find that information in the course.",
            []
        )

    context = build_context(retrived_chunks)
    prompt = f"""Context:
    {context}

Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.3,
        messages=[
            {
                "role": "system",
                "content": (
                   "You are an AI assistant for programming courses. "
                    "Answer ONLY using the retrieved context. "
                    "Do not invent facts. "
                    "Combine information from multiple chunks when appropriate. "
                    "If the answer cannot be found in the context, reply exactly: "
                    "'I couldn't find that information in the course.'" 
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response.choices[0].message.content
    return answer, retrived_chunks


def main():

    while True:
        question = input("Ask a question ('or type exit'): ")
        if question.lower() == "exit":
            break

        answer, sources = answer_question(question)
        print("\n" + "=" * 80)
        print("ANSWER")
        print("=" * 80)
        print(answer)

        print("\n" + "=" * 80)
        print("SOURCES")
        print("=" * 80)

        for source in sources:

            print(f"Lecture : {source['title']}")
            print(f"File     : {source['source_file']}")
            print(f"Time     : {source['start']} - {source['end']}")
            print("-" * 80)


if __name__ == "__main__":
    main()


