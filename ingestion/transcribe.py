import os
import json
import av
from faster_whisper import WhisperModel

VIDEO_DIR = "data/videos"
OUTPUT_DIR = "data/transcripts"

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = WhisperModel(
    "base",
    device = "cpu",
    compute_type= "int8",
    cpu_threads=8
)

def transcribe_file(file_path, title):
    segments, info = model.transcribe(file_path, beam_size=5)

    chunks = []
    for segment in segments:
        chunks.append({
            "title": title,
            "start": segment.start,
            "end": segment.end,
            "duration": segment.end - segment.start,
            "text": segment.text.strip()
        })
    return {
        "chunks": chunks
    }


def save_transcripts(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )

def main():
    supported_formats = (
        ".mp3",
        ".mp4",
        ".wav",
        ".mkv",
        ".webm",
        ".mov"
    )

    files=[
        f for f in os.listdir(VIDEO_DIR)
        if f.lower().endswith(supported_formats)
    ]

    print(f"Found {len(files)} files")

    for filename in files:

        file_path = os.path.join(
            VIDEO_DIR,
            filename
        )

        base_name = os.path.splitext(filename)[0]

        output_file = os.path.join(
            OUTPUT_DIR,
            f"{base_name}.json"
        )

        if os.path.exists(output_file):
            continue

        print(f"Processing {filename}")

        try:
            transcipt = transcribe_file(file_path, base_name)

        except av.error.InvalidDataError as e:
            print(f"Failed to process {filename}: {e}")
            continue

        save_transcripts(transcipt, output_file)

        print(f"Saved -> {output_file}")

if __name__ == "__main__":
    main()