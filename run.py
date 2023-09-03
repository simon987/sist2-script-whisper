from time import time

import typer
import os
import subprocess
from sist2 import Sist2Index, print_progress

MODELS = {
    "base": "whisper.cpp/models/ggml-base-q4_1.bin",
    "small": "whisper.cpp/models/ggml-small-q4_1.bin",
    "large": "whisper.cpp/models/ggml-large-q4_1.bin"
}


def whisper_stt(input_audio: str, num_threads: int, model: str):
    wav_path = "/tmp/whisper-tmp.wav"
    try:
        os.remove(wav_path)
    except:
        pass

    subprocess.run([
        "ffmpeg",
        "-hide_banner", "-loglevel", "panic",
        "-i", input_audio,
        "-ar", "16000",
        wav_path
    ])

    subprocess.run([
        "whisper.cpp/main",
        "-t", str(num_threads),
        "-m", MODELS[model],
        "-f", wav_path,
        "--output-txt",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    with open(wav_path + ".txt") as f:
        text = f.read()

    return text


def main(index_file: str, num_threads: int = 8, color: str = "#51da4c", tag: bool = True, model: str = "base"):
    if model not in MODELS:
        raise ValueError(f"Model {model} not found. Possible values: {MODELS.keys()}")

    index = Sist2Index(index_file)

    tag_value = f"whisper.{color}"

    # Only consider documents that were modified since the last run of this script
    whisper_version = index.get("whisper_version", default=0)

    where = f"((SELECT name FROM mime WHERE id=document.mime) LIKE 'audio/%' " \
            f"OR (SELECT name FROM mime WHERE id=document.mime) LIKE 'video/%') AND version > {whisper_version}"

    total = index.document_count(where)
    done = 0

    for doc in index.document_iter(where=where):
        start = time()
        text = whisper_stt(doc.path, num_threads, model)

        doc.json_data["content"] = text

        if tag:
            if "tags" not in doc.json_data:
                doc.json_data["tag"] = [tag_value]
            else:
                doc.json_data["tag"] = list(filter(lambda t: not t.startswith("whisper."), doc.json_data["tag"])) \
                    .append(tag_value)

        print(f"Performed STT for {doc.rel_path} ({time() - start:.2f}s)")
        index.update_document(doc)

        done += 1
        print_progress(done=done, count=total)

    index.set("whisper_version", index.versions[-1].id)

    print("Done!")

    index.sync_tag_table()
    index.commit()


if __name__ == "__main__":
    typer.run(main)
