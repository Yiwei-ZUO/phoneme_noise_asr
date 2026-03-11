import argparse
import csv
import os
import random
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--corpus_dir", required=True, help="Path to the downloaded corpus")
parser.add_argument("--lang", default="en", help="Language code (e.g., en, fr)")
parser.add_argument("--n_samples", type=int, default=100, help="Number of audio samples to extract")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
args = parser.parse_args()

# Setup output paths
wav_dir = f"data/raw/{args.lang}/wav"
out_tsv = f"data/raw/{args.lang}/transcripts.tsv"
tmp_tsv = f"{out_tsv}.tmp"

os.makedirs(wav_dir, exist_ok=True)

# Load and filter valid rows from the corpus
with open(f"{args.corpus_dir}/ss-corpus-{args.lang}.tsv", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    valid_rows = [row for row in reader if row["transcription"].strip()]

# Sample rows with a fixed random seed for reproducibility
random.seed(args.seed)
sampled_rows = random.sample(valid_rows, args.n_samples)

# 3. Process audio and write manifest into a temporary file first
with open(tmp_tsv, "w", encoding="utf-8") as fout:
    for i, row in enumerate(sampled_rows, 1):
        stem = row["audio_file"].replace(".mp3", "")
        in_audio = f"{args.corpus_dir}/audios/{row['audio_file']}"
        out_audio = f"{wav_dir}/{stem}.wav"

        # Convert audio to 16kHz mono
        subprocess.run([
            "ffmpeg", "-i", in_audio,
            "-ar", "16000", "-ac", "1", out_audio,
            "-y", "-loglevel", "error"
        ], check=True)

        # Write transcript to temporary TSV
        transcript = row["transcription"].strip()
        fout.write(f"{stem}\t{transcript}\n")
        print(f"[{i}/{args.n_samples}] Processed {stem}")

# Atomic write: rename tmp file to final destination ONLY if everything succeeds
os.replace(tmp_tsv, out_tsv)
print(f"Success! Atomically saved {args.n_samples} transcripts to {out_tsv}")