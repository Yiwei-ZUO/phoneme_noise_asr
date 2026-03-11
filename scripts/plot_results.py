import argparse
import json
import os
import matplotlib.pyplot as plt


def plot(metrics_dir, output_path, langs, snr_levels):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load all metrics by explicitly constructing paths from parameters
    # This avoids folder scanning and is fully driven by params.yaml
    all_data = []
    for lang in langs:
        for snr in snr_levels:
            snr_str = int(snr) if snr == int(snr) else snr
            fpath = os.path.join(metrics_dir, f"{lang}_{snr_str}db.json")
            if os.path.exists(fpath):
                with open(fpath) as f:
                    all_data.append(json.load(f))
            else:
                print(f"Warning: {fpath} not found, skipping.")

    fig, ax = plt.subplots(figsize=(10, 6))
    all_per = []

    for lang in langs:
        results = [r for r in all_data if r["lang"] == lang and r["snr_db"] is not None]
        results.sort(key=lambda x: float(x["snr_db"]))
        if not results:
            continue

        snr_vals = [r["snr_db"] for r in results]
        per_vals = [r["per"] for r in results]
        all_per.append((snr_vals, per_vals))
        ax.plot(snr_vals, per_vals, marker="o", label=lang)

    # Cross-language mean
    if len(all_per) > 1:
        mean_per = [sum(p[i] for _, p in all_per) / len(all_per)
                    for i in range(len(all_per[0][1]))]
        ax.plot(all_per[0][0], mean_per, marker="s",
                linestyle="--", color="black", label="mean")

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("PER")
    ax.set_title("Phoneme Error Rate vs Noise Level")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    # Atomic write
    ext = os.path.splitext(output_path)[1].lstrip('.')
    tmp_path = output_path + ".tmp"
    plt.savefig(tmp_path, format=ext)
    os.replace(tmp_path, output_path)
    print(f"Figure saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", required=True)
    parser.add_argument("--output",      required=True)
    parser.add_argument("--langs",       required=True, nargs="+")
    parser.add_argument("--snr-levels",  required=True, nargs="+", type=float)
    args = parser.parse_args()
    plot(args.metrics_dir, args.output, args.langs, args.snr_levels)