#!/usr/bin/env python3
"""
Stage 2: Add noise to audio files at different SNR levels.
Input:  data/manifests/{lang}/clean.jsonl
Output: data/manifests/{lang}/noisy_{snr}db.jsonl
"""

import argparse
import hashlib
import json
import os
import numpy as np
import soundfile as sf

def add_noise(signal, snr_db, rng):
    """Adds white noise based on the desired SNR."""
    signal_power = np.mean(signal ** 2)
    # Convert SNR from dB to linear scale
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = rng.normal(loc=0.0, scale=np.sqrt(noise_power), size=signal.shape)
    return signal + noise

def add_noise_to_file(input_wav, output_wav, snr_db, seed=None):
    """Processes a single audio file."""
    signal, sr = sf.read(input_wav)
    if signal.ndim != 1:
        raise ValueError(f"Only mono audio is supported, but {input_wav} has {signal.ndim} channels.")
    
    rng = np.random.default_rng(seed)
    noisy_signal = add_noise(signal, snr_db, rng)
    sf.write(output_wav, noisy_signal, sr)

def main(clean_manifest, lang, snr_db, output_dir, output_manifest):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_manifest), exist_ok=True)

    # Following the principle of Atomic Creation
    tmp_path = output_manifest + ".tmp"

    with open(clean_manifest, encoding="utf-8") as fin, \
         open(tmp_path, "w", encoding="utf-8") as fout:

        for line in fin:
            record = json.loads(line)

            # Strictly consuming manifest instead of scanning folders
            input_wav = record["wav_path"]
            fname = os.path.basename(input_wav)
            output_wav_path = os.path.join(output_dir, fname)

            # Ensure seed reproducibility based on utt_id and snr
            seed_str = f"{record['utt_id']}_{snr_db}"
            seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**32)
            
            # Process audio
            add_noise_to_file(input_wav, output_wav_path, snr_db, seed=seed)

            # Update record for the new manifest
            new_record = record.copy()
            # Ensure the path saved is a relative path
            new_record["wav_path"] = os.path.relpath(output_wav_path, start=os.getcwd())
            new_record["snr_db"] = snr_db
            
            fout.write(json.dumps(new_record, ensure_ascii=False) + "\n")

    # Atomic rename: manifest is only updated if all items are processed
    os.replace(tmp_path, output_manifest)
    print(f"Successfully created noisy manifest: {output_manifest}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inject noise into clean audio manifest.")
    parser.add_argument("--clean-manifest",  required=True)
    parser.add_argument("--lang",            required=True)
    parser.add_argument("--snr-db",          required=True, type=float)
    parser.add_argument("--output-dir",      required=True)
    parser.add_argument("--output-manifest", required=True)
    args = parser.parse_args()

    main(args.clean_manifest, args.lang, args.snr_db,
         args.output_dir, args.output_manifest)