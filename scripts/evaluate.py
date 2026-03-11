#!/usr/bin/env python3
"""
Stage 4: Compute Phoneme Error Rate (PER).
Input:  data/preds/{lang}/preds_{snr}.jsonl
Output: results/metrics/{lang}_{snr}.json
"""

import argparse
import json
import os
import unicodedata


def normalize_phonemes(phon_str):
    # Remove stress markers and length marker
    phon_str = phon_str.replace("ˈ", "").replace("ˌ", "").replace("ː", "")
    # Normalize unicode
    phon_str = unicodedata.normalize("NFD", phon_str)
    # Remove diacritics
    phon_str = "".join(c for c in phon_str if unicodedata.category(c) != "Mn")
    phon_str = phon_str.lower().strip()
    # Split into individual characters, ignoring spaces
    tokens = list(phon_str.replace(" ", ""))
    return " ".join(tokens)


def edit_distance(ref, hyp):
    """Standard dynamic programming edit distance."""
    r, h = ref.split(), hyp.split()
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = 1 + min(d[i-1][j],
                                   d[i][j-1],
                                   d[i-1][j-1])
    return d[len(r)][len(h)]


def compute_per(pred_manifest):
    total_errors = 0
    total_phones = 0

    with open(pred_manifest, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            ref = normalize_phonemes(record["ref_phon"])
            hyp = normalize_phonemes(record["pred_phon"])
            n = len(ref.split())
            if n == 0:
                continue
            total_errors += edit_distance(ref, hyp)
            total_phones += n

    per = total_errors / total_phones if total_phones > 0 else 0.0
    return round(per, 4)


def main(pred_manifest, output_path, lang, snr_db):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    per = compute_per(pred_manifest)

    result = {
        "lang":   lang,
        "snr_db": snr_db,
        "per":    per
    }

    # Atomic write
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    os.replace(tmp_path, output_path)

    print(f"PER ({lang}, SNR={snr_db}dB): {per:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-manifest", required=True)
    parser.add_argument("--output",        required=True)
    parser.add_argument("--lang",          required=True)
    parser.add_argument("--snr-db",        required=True)
    args = parser.parse_args()
    main(args.pred_manifest, args.output, args.lang, args.snr_db)