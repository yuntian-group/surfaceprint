# Official Implementation of SurfacePrint: Typographic Fingerprints of LLMs

Short guide for running the scripts in this repo. Always run from the **repository root**.

## 1. Build the Environment

The repo ships an Apptainer (Singularity) definition under `container/`. Build the image once with:

```bash
bash container/build.sh
```

This produces `container/venv/surfaceprint.sif` (override with `DEST_DIR=/path bash container/build.sh`). The image contains PyTorch 2.10 + CUDA 12.8, Flash Attention 3, vLLM 0.19, and HuggingFace `transformers`.

Requirements on the host:

- `apptainer` with `--fakeroot` (or root) build privileges
- A CUDA-capable GPU and matching NVIDIA driver at run time
- ~30 GB free in `/tmp` for the build cache

## 2. Setup

The repo ships a template `init.sh` at the root. Open it once and fill in the placeholder values (search for `FILL_ME`):

- API keys: `HF_TOKEN` (only required if you pull gated HF models).
- Optionally: HuggingFace / dataset cache directories, extra `--bind` mounts for shared model-weight or scratch paths, and `module load apptainer` if your cluster needs it.
- `CONTAINER_IMAGE` defaults to `container/venv/surfaceprint.sif` from step 1.

Then use it three ways:

```bash
./init.sh                          # interactive shell inside the container
./init.sh python my_script.py      # one-shot command inside the container
source init.sh && run_in_apptainer bash scripts/<some>.sh   # for batch/Slurm
```

## 3. Dataset

Most scripts expect the merged dataset at:

```
dataset/merged_dataset
```

Download the prebuilt archive from Google Drive:

[merged_dataset.tar.zst TO BE RELEASED AFTER REVIEW]()

Place the downloaded `merged_dataset.tar.zst` file under `dataset/` and extract it so the final layout is `dataset/merged_dataset/`:

```bash
mkdir -p dataset
mv /path/to/merged_dataset.tar.zst dataset/
tar --use-compress-program=unzstd -xf dataset/merged_dataset.tar.zst -C dataset/
# or, equivalently:
# zstd -d dataset/merged_dataset.tar.zst -c | tar -xf - -C dataset/
```

After extraction you should see `dataset/merged_dataset/{train,val,test}` and `dataset/merged_dataset/dataset_dict.json`.

Override the location with the `DATASET_PATH` env var if needed.

## 4. Train ASCII Detectors (`scripts/`)


| What it trains                                                       | Command                                                                                                                     |
| -------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| One detector per model (human vs. each model), whitespace-normalized | `bash scripts/train_single_model_all_data.sh`                                                                               |
| Same, but on raw (un-normalized) text                                | `bash scripts/train_single_model_all_data_raw_normalize.sh`                                                                 |
| Unified human-vs-instruct detector, whitespace-normalized            | `bash scripts/train_unified_ascii_detector_all_data.sh`                                                                     |
| Unified detector, raw text                                           | `bash scripts/train_unified_ascii_detector_all_data_raw_normalization.sh`                                                   |
| Unified **binary** ASCII features (depth-3 / depth-5 trees)          | `bash scripts/train_unified_binary_ascii_detector_all_data_raw_tree_depth3.sh` / `..._depth5.sh` (also `_ws_norm` variants) |
| Unified **full** ASCII features (depth-3 / depth-5 trees)            | `bash scripts/train_unified_full_ascii_detector_all_data_raw_tree_depth3.sh` / `..._depth5.sh`                              |


Outputs go to `models/...` and per-run metrics JSONs.

## 5. Statistical Baselines (`scripts/statistical/`)

Run one baseline (Likelihood, LogRank, Entropy, LRR, Fast-DetectGPT, DNA-DetectLLM, LastDE++) per target model:

```bash
bash scripts/statistical/run_baseline_statistical_<model>.sh
```

Examples:

```bash
bash scripts/statistical/run_baseline_statistical_gpt_4o_mini.sh
bash scripts/statistical/run_baseline_statistical_llama3_1_8b_instruct.sh
bash scripts/statistical/run_baseline_statistical_claude_sonnet_4.sh
```

Outputs land in `results/baseline_statistical/<model>/`.

## 6. Generalization Experiments (`scripts/generalization/`)


| Setup                                           | Command                                                                                   |
| ----------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Train on one model, test on all others          | `bash scripts/generalization/run_generalization_detector_one_to_all.sh`                   |
| Train on all-others, test on the held-out model | `bash scripts/generalization/run_generalization_detector_all_other_to_one.sh`             |
| Open-source → closed-source transfer            | `bash scripts/generalization/run_generalization_detector_open_source_to_closed_source.sh` |


Each has a `..._raw.sh` variant that skips whitespace normalization.

## 7. Common Env Overrides


| Var                                           | Default                                          | Used by                 |
| --------------------------------------------- | ------------------------------------------------ | ----------------------- |
| `DATASET_PATH`                                | `dataset/merged_dataset`                         | most train/eval scripts |
| `OUTPUT_DIR`                                  | per-script default under `models/` or `results/` | all train scripts       |
| `MODEL` / `SCORING_MODEL` / `REFERENCE_MODEL` | per-script                                       | statistical baselines   |
| `CUDA_VISIBLE_DEVICES`                        | `0`                                              | GPU-using scripts       |


