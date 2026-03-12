#!/usr/bin/env python3
"""
Stage 3: Automatic Speech Recognition (ASR) - Phoneme Prediction.
Input:  data/manifests/{lang}/ directory with noisy manifests
Output: data/preds/{lang}/ directory with prediction manifests
"""

# Fix for OpenMP conflict on macOS with Apple Silicon
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import platform
import shutil
import subprocess
import yaml
import torch
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


def setup_espeak():
    if os.environ.get("PHONEMIZER_ESPEAK_LIBRARY"):
        return
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(["brew", "--prefix", "espeak-ng"],
                                    capture_output=True, text=True, check=False)
            if result.returncode == 0:
                lib = os.path.join(result.stdout.strip(), "lib", "libespeak-ng.dylib")
                if os.path.exists(lib):
                    os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = lib
                    return
        except FileNotFoundError:
            pass
    if not shutil.which("espeak-ng") and not os.environ.get("PHONEMIZER_ESPEAK_LIBRARY"):
        print("Warning: espeak-ng not found. Please ensure it is installed and in PATH.")

setup_espeak()

MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"


def load_model():
    print(f"Loading model {MODEL_ID}...")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
    model.eval()
    return processor, model


def predict_phonemes(wav_path, processor, model):
    signal, sr = sf.read(wav_path)
    assert sr == 16000, f"Expected 16kHz, got {sr}"
    assert signal.ndim == 1, "Expected mono audio"

    inputs = processor(signal, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    return processor.decode(predicted_ids[0])


def process_manifest(input_manifest, output_manifest, processor, model):
    os.makedirs(os.path.dirname(output_manifest), exist_ok=True)
    tmp_path = output_manifest + ".tmp"

    with open(input_manifest, encoding="utf-8") as fin, \
         open(tmp_path, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            record = json.loads(line)
            pred_phon = predict_phonemes(record["wav_path"], processor, model)
            new_record = record.copy()
            new_record["pred_phon"] = pred_phon
            fout.write(json.dumps(new_record, ensure_ascii=False) + "\n")
            print(f"[{i+1}] {record['utt_id']} → {pred_phon}")

    os.replace(tmp_path, output_manifest)
    print(f"Predictions written to {output_manifest}")


def main(input_dir, output_dir, params_file="params.yaml"):
    with open(params_file) as f:
        params = yaml.safe_load(f)
    snr_levels = params["snr_levels"]

    os.makedirs(output_dir, exist_ok=True)

    # Load model once for all SNR levels
    processor, model = load_model()

    for snr in snr_levels:
        snr_str = int(snr) if snr == int(snr) else snr
        input_manifest = os.path.join(input_dir, f"noisy_{snr_str}db.jsonl")
        if not os.path.exists(input_manifest):
            print(f"Warning: {input_manifest} not found, skipping...")
            continue
        output_manifest = os.path.join(output_dir, f"preds_{snr_str}db.jsonl")
        print(f"\nProcessing SNR {snr_str}dB: {input_manifest}...")
        process_manifest(input_manifest, output_manifest, processor, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir",  required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--params",     default="params.yaml")
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.params)