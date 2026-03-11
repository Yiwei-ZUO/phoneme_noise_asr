#!/usr/bin/env python3
"""
Stage 3: Automatic Speech Recognition (ASR) - Phoneme Prediction.

For each utterance in the input manifest, this stage:
1. Reads the audio file (.wav, 16kHz mono)
2. Feeds it to the pre-trained model facebook/wav2vec2-lv-60-espeak-cv-ft
3. Decodes the model output into a phoneme sequence
4. Writes a new manifest with the predicted phonemes added as 'pred_phon'

Input:  data/manifests/{lang}/clean.jsonl or noisy_{snr}db.jsonl
Output: data/preds/{lang}/preds_clean.jsonl or preds_{snr}db.jsonl
"""

import argparse
import json
import os
import torch
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import platform
import subprocess
import shutil

def setup_espeak():
    if os.environ.get("PHONEMIZER_ESPEAK_LIBRARY"):
        return
    # Dynamic library loading for espeak-ng to ensure phonemizer works across environments
    if platform.system() == "Darwin":
        try:
            # Use subprocess to dynamically locate espeak-ng via Homebrew
            result = subprocess.run(["brew", "--prefix", "espeak-ng"], 
                                   capture_output=True, text=True, check=False)
            if result.returncode == 0:
                lib = os.path.join(result.stdout.strip(), "lib", "libespeak-ng.dylib")
                if os.path.exists(lib):
                    os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = lib
                    return
        except FileNotFoundError:
            # 'brew' command not found, likely a Linux server or non-Homebrew environment
            pass
    
    # Fallback check for Linux/Standard environments
    if not shutil.which("espeak-ng") and not os.environ.get("PHONEMIZER_ESPEAK_LIBRARY"):
        print("Warning: espeak-ng not found. Please ensure it is installed and in PATH.")
        
setup_espeak()   

MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"

def load_model():
    print(f"Loading model {MODEL_ID}...")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model     = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
    model.eval()
    return processor, model

def predict_phonemes(wav_path, processor, model):
    signal, sr = sf.read(wav_path)
    # Model requires 16kHz mono
    assert sr == 16000, f"Expected 16kHz, got {sr}"
    assert signal.ndim == 1, "Expected mono audio"

    inputs = processor(signal, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription

def main(input_manifest, output_manifest):
    os.makedirs(os.path.dirname(output_manifest), exist_ok=True)
    tmp_path = output_manifest + ".tmp"

    processor, model = load_model()

    with open(input_manifest, encoding="utf-8") as fin, \
         open(tmp_path, "w", encoding="utf-8") as fout:

        for i, line in enumerate(fin):
            record = json.loads(line)
            pred_phon = predict_phonemes(record["wav_path"], processor, model)

            # Add prediction to record
            new_record = record.copy()
            new_record["pred_phon"] = pred_phon
            fout.write(json.dumps(new_record, ensure_ascii=False) + "\n")
            print(f"[{i+1}] {record['utt_id']} → {pred_phon}")

    # Atomic write
    os.replace(tmp_path, output_manifest)
    print(f"Predictions written to {output_manifest}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest",  required=True)
    parser.add_argument("--output-manifest", required=True)
    args = parser.parse_args()
    main(args.input_manifest, args.output_manifest)