"""
Stage 1: Build the clean manifest.
Input:  data/raw/{lang}/wav/ + data/raw/{lang}/transcripts.tsv
Output: data/manifests/{lang}/clean.jsonl (line-based JSONL)
"""

import argparse
import hashlib
import json
import os
import subprocess
import soundfile as sf

def get_md5(path):
    """Generate MD5 checksum for file traceability."""
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def text_to_phonemes(text, lang):
    """Convert raw text to phoneme sequence using espeak-ng."""
    # Using subprocess for cross-platform compatibility via pixi/conda
    result = subprocess.run(
        ["espeak-ng", "-v", lang, "-q", "--ipa", text],
        capture_output=True, text=True
    )
    # Clean up output: remove newlines and extra spaces
    return result.stdout.strip().replace("\n", " ")

def make_manifest(wav_dir, transcript_file, lang, output_path):
    transcripts = {}
    if not os.path.exists(transcript_file):
        raise FileNotFoundError(f"Input transcript not found: {transcript_file}")

    with open(transcript_file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                transcripts[parts[0]] = parts[1]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Atomic creation: Write to a temporary file first
    tmp_path = output_path + ".tmp"

    with open(tmp_path, "w", encoding="utf-8") as out:
        for stem in sorted(transcripts.keys()):
            wav_path = os.path.join(wav_dir, f"{stem}.wav")

            if not os.path.exists(wav_path):
                continue

            ref_text = transcripts[stem]

            # Filtering logic (ensure clean data for phonemization)
            if "<" in ref_text or ">" in ref_text:
                continue
            if lang == "en" and not all(ord(c) < 128 for c in ref_text):
                continue

            # Process audio metadata
            signal, sr = sf.read(wav_path)
            duration = len(signal) / sr
            md5 = get_md5(wav_path)
            ref_phon = text_to_phonemes(ref_text, lang)
            
            utt_id = f"{lang}_{stem}"
            
            # Ensure the path in manifest is relative 
            rel_wav_path = os.path.relpath(wav_path, start=os.getcwd())

            record = {
                "utt_id": utt_id,
                "lang": lang,
                "wav_path": rel_wav_path,
                "ref_text": ref_text,
                "ref_phon": ref_phon,
                "sr": sr,
                "duration_s": round(duration, 3),
                "snr_db": None,
                "audio_md5": md5
            }
            # Write the record as a JSON line to the temporary manifest
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Finalize Atomic Write: only rename if everything succeeded
    os.replace(tmp_path, output_path)
    print(f"Manifest successfully created at {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a phoneme ASR manifest.")
    parser.add_argument("--wav-dir",      required=True, help="Path to wav files")
    parser.add_argument("--transcripts",  required=True, help="Path to TSV transcript file")
    parser.add_argument("--lang",         required=True, help="Language code")
    parser.add_argument("--output",       required=True, help="Path to output .jsonl manifest")
    args = parser.parse_args()

    make_manifest(args.wav_dir, args.transcripts, args.lang, args.output)