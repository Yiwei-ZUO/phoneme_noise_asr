# Phoneme ASR Robustness to Noise

A fully reproducible DVC pipeline for evaluating phoneme ASR robustness to noise across multiple languages and SNR levels.

## Setup
```bash
pixi install
```

## Data

This pipeline expects a corpus in the following format:
- A TSV file named `ss-corpus-{lang}.tsv` with columns `audio_file` and `transcription`
- An `audios/` folder containing the corresponding `.mp3` files

The pipeline was tested with:
- English: [Common Voice Spontaneous Speech 1.0](https://datacollective.mozillafoundation.org/datasets/cmihqzerk023co20749miafhq)
- French: [Common Voice Spontaneous Speech 2.0](https://datacollective.mozillafoundation.org/datasets/cmj8u48ad0025nxzp2lfaluce)

Replace `--corpus_dir` with your own local path when running `download_data.py`.

## Run the pipeline

### Step 1: Prepare raw data
```bash
pixi run python scripts/download_data.py \
  --corpus_dir /path/to/your/corpus \
  --lang en
```

### Step 2: Run full pipeline
```bash
pixi run dvc repro
```

## Add a new language

Edit `params.yaml` and add the language code:
```yaml
langs:
  - en
  - fr
```

Then run:
```bash
pixi run dvc repro
```

DVC will only recompute the stages that are affected by the change.

## Pipeline stages

| Stage | Script | Description |
|-------|--------|-------------|
| `make_manifest` | `make_manifest.py` | Build clean manifest from raw audio |
| `add_noise` | `add_noise.py` | Add noise at 10 SNR levels |
| `run_asr` | `run_asr.py` | Run wav2vec2 phoneme recognition |
| `evaluate` | `evaluate.py` | Compute Phoneme Error Rate (PER) |
| `plot_results` | `plot_results.py` | Plot PER vs SNR curve |

## Requirements

- [pixi](https://prefix.dev/docs/pixi/overview)
- [espeak-ng](https://github.com/espeak-ng/espeak-ng) (macOS: `brew install espeak-ng`)
- [ffmpeg](https://ffmpeg.org) (managed by pixi)