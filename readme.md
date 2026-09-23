# course-video-rag-assistant

A RAG-based Q&A system for video course content — retrieves relevant transcript chunks via FAISS similarity search and generates grounded answers using OpenAI embeddings and GPT-4.1-mini.

## Overview

Transcribes video files via faster-whisper, chunks the transcripts with overlap, embeds them with OpenAI's embedding model, indexes them in FAISS, retrieves relevant chunks for a query, and generates a grounded answer with GPT-4.1-mini.


## Architecture

transcribe.py -> Transcribes video files and saves the transcripts to the transcript folder.
chunk.py -> Splits transcripts into ~500-character chunks with 10% overlap and saves them to the chunks folder.
embed.py -> Creates embeddings for each chunk, attaches them to the chunk data, and saves the result to the embeddings folder.
build_index.py -> Gathers chunks from every video's chunk file, builds a single FAISS index across all of them, and saves the chunk metadata to metadata.json.
search.py -> Retrieves the top relevant chunks for a query (filtered by similarity threshold) and prints them with their scores.
answer.py -> Retrieves relevant chunks for a query and generates a grounded answer with GPT-4.1-mini, returning the sources used.

## Design Decisions

Relevance threshold (0.40 cosine similarity): Originally, retrieval always returned the top 5 chunks regardless of relevance, and the LLM alone decided whether the context was sufficient to answer — meaning genuinely irrelevant chunks were passed into every prompt. I added a similarity threshold so retrieval itself filters out irrelevant chunks before they ever reach the LLM. The 0.40 cutoff was chosen from real testing: a clearly on-topic question ("what is CSS?") scored 0.48+ on its weakest true match, while a clearly off-topic question ("capital of France") scored below 0.40 on every candidate — 0.40 sits in that gap. One known limitation remains: even when relevant-scoring chunks are returned, the LLM can still correctly refuse to answer if none of them explicitly address the specific question asked — verified with a "tags vs attributes" query, where the top 5 retrieved chunks scored above threshold but none explicitly covered the distinction, so the refusal was accurate, not a retrieval failure.

Chunk size (500 characters) with 10% overlap: I moved from the original 1000-character chunks down to 500 after testing, because smaller chunks let a query match content that's tightly focused on the topic asked, rather than retrieving a larger chunk where the relevant sentence is buried among unrelated surrounding content. There's a real tradeoff at the extremes, though — too small (e.g. 100 characters) and a chunk can no longer hold a complete idea or sentence, so even a chunk that scores a strong similarity match returns a fragment the LLM can't actually use to answer well; it also means far more chunks overall, driving up embedding API calls and retrieval overhead. 500 characters balances focus against coherence. Since chunking is a hard cut and doesn't respect sentence or idea boundaries, a thought can still span two chunks — I added 10% overlap (the tail of one chunk repeated at the start of the next) so a boundary-straddling idea isn't fully lost to either side. The first chunk in each file has no overlap, since nothing precedes it.

Hash-based re-embedding check (embed.py): The original approach skipped re-embedding a file if an output file with that name already existed — but the filename is based on the source video, not the chunk parameters, so if I changed chunk.py's MAX_CHAR and reran the pipeline, the stale embeddings from the old chunk size were silently kept and the check never detected that the underlying content had changed. I fixed this by computing an MD5 hash of all a file's chunk text (joined with a separator so chunk boundaries aren't lost in the hash) and storing it alongside the embeddings. On each run, the current chunks' hash is compared against the stored one — if they match, the content is genuinely unchanged and re-embedding is skipped; if they differ (or no valid hash exists yet), the file is re-embedded. This avoids the unnecessary API cost of re-embedding files that haven't changed, while still catching real content changes that a filename-only check would miss.

FAISS IndexFlatIP with normalized vectors: I chose inner-product search over normalized vectors (equivalent to cosine similarity) instead of L2 (Euclidean) distance. L2 distance is unbounded above — a distance of 0 means identical vectors, but there's no fixed upper bound, so a raw L2 number can only be judged relative to other results in the same search, not against a fixed cutoff. Cosine similarity is bounded (roughly 0 to 1 in practice for real text embeddings, with 1 meaning identical direction), which makes it possible to set a fixed, interpretable relevance threshold — the 0.40 cutoff described above wouldn't be meaningful against unbounded L2 distances the same way.

## Known Limitations

No formal threshold validation: The 0.40 similarity cutoff was chosen from a small manual test (one clearly on-topic and one clearly off-topic question), not a systematic evaluation. It's possible the optimal threshold differs for other question types — e.g., a question with a very specific, rare match might legitimately score lower than 0.40 and get incorrectly filtered out, or vague/broad questions might match many chunks above 0.40 without genuinely being answerable. A proper evaluation would need a labeled test set of many questions with known-good answers to measure precision/recall at different thresholds.

Overlap/timestamp mismatch: Each chunk after the first carries ~10% of the previous chunk's trailing text for continuity, but the chunk's stored start timestamp reflects where its own new content begins — not where the carried-over overlap text originally occurred. So for chunks with overlap, the displayed time range is slightly narrower than the actual span of text shown, by roughly the overlap length. This doesn't affect retrieval or answer quality, only the precision of the displayed source timestamp.

Two distinct "no answer" paths: The system can return "I couldn't find that information" for two different reasons, and the sources list is what distinguishes them. If retrieval finds no chunks above the 0.40 threshold, the fallback fires immediately with an empty sources list — genuinely nothing relevant exists in the index for that query. But even when relevant-scoring chunks ARE retrieved (sources list populated), the LLM can still refuse to answer if none of those chunks explicitly contain the specific fact asked — verified with a "difference between tags and attributes" query, where the top 5 retrieved chunks scored above threshold but none explicitly addressed that distinction, so the refusal was accurate rather than a retrieval bug. A user or caller needs to check whether sources is empty to know which case occurred.

Full index rebuild on every run: build_index.py always builds a brand-new FAISS index from all embeddings, rather than incrementally adding only new vectors. At 555 vectors this is fast and not a real problem, but it wouldn't scale efficiently to a much larger dataset, where rebuilding from scratch every time would become expensive.

## Setup / How to Run
 1. Install dependencies:
   pip install faster-whisper openai faiss-cpu numpy python-dotenv
 2. Create a .env file in the project root with your OpenAI API key:
   OPENAI_API_KEY=your_key_here
 3. Add video/audio files to data/videos/, then run the pipeline in order:
   python ingestion/transcribe.py
   python ingestion/chunk.py
   python indexing/embed.py
   python indexing/build_index.py
   python retrieval/search.py   # for raw retrieval testing
   python answer.py   # for full Q&A with generated answers
