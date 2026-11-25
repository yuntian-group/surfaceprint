# Experiments

Run these commands from the repository root after installing the dependencies
and extracting the dataset as described in the [README](../README.md).
`DATASET_PATH` selects the dataset directory; `OUTPUT_DIR` selects where each
shell script writes its models and metrics.

## Per-model detection

Train a separate human-versus-model classifier for each generator:

```bash
bash scripts/train_per_model_raw.sh
bash scripts/train_per_model_ws.sh
```

These scripts use a depth-6 decision tree. For a random forest for one generator:

```bash
python train_merged_ascii_detector.py \
  --dataset-path dataset/merged_dataset \
  --include-source human gpt-4o-mini \
  --label-mode source --target-source gpt-4o-mini \
  --classifier forest --normalize-mode raw \
  --output models/gpt-4o-mini.pkl \
  --metrics-json models/gpt-4o-mini.metrics.json
```

The forest uses 300 trees, balanced class weights, random seed 0, and
scikit-learn's default tree-growth settings. `--max-depth` and
`--min-samples-leaf` configure the single-tree classifier. The default evaluation
split is `test`; `--val-split val` selects the validation split.

## Shallow decision trees

Train a depth-3 tree over the full 97-feature vector or its binary variant:

```bash
bash scripts/train_unified_full_ascii_detector_all_data_raw_tree_depth3.sh
bash scripts/train_unified_binary_ascii_detector_all_data_raw_tree_depth3.sh
```

The corresponding `depth5` scripts train depth-5 trees. `full` uses feature
rates and densities, `binary` uses 97 presence indicators, and `basic` uses
45 character, word, and pattern indicators. Scripts ending in `_ws_norm.sh`
remove ASCII spaces, tabs, carriage returns, and newlines before extraction.

## Model attribution

Predict which instruction-tuned model generated a response:

```bash
bash scripts/train_authorship_raw.sh
bash scripts/train_authorship_ws.sh
```

These train a multiclass random forest over the 17 model sources. The output
class is a generator identifier; human responses are excluded.

## Transfer to unseen models

```bash
bash scripts/generalization/run_generalization_detector_one_to_all_raw.sh
bash scripts/generalization/run_generalization_detector_all_other_to_one_raw.sh
bash scripts/generalization/run_generalization_detector_open_source_to_closed_source_raw.sh
```

One-to-all trains on one generator and tests on each generator. Leave-one-out
trains on the other generators and tests on the held-out generator.
Open-to-closed trains on open-weight models and tests on API models. Human
responses provide the negative class in each experiment.

The scripts without `_raw` remove ASCII whitespace before extracting features.
The experiment plans in [`configs/generalization`](../configs/generalization/)
list the training and test sources. Each plan runs with the classifier,
sampling settings, and seed specified by its shell script and
[`train_generalization_detector.py`](../train_generalization_detector.py).

## Inspect learned features

```bash
python scripts/analyze_ascii_features.py --model models/gpt-4o-mini.pkl
python scripts/analyze_ascii_forest_features.py --model models/gpt-4o-mini.pkl --split-usage
```

The first command reports feature importance. The second reports how often
each feature is used across the forest.

## GPU baselines

The statistical and embedding baselines require PyTorch and Transformers.
The Apptainer definition includes CUDA 12.8, PyTorch 2.10, Transformers 5.5,
and Flash Attention 3 for the embedding baseline.

Build on Linux with Apptainer and enough temporary space for the CUDA image:

```bash
bash container/build.sh
```

Use a Hugging Face account with access to the selected model repositories and
set `HF_TOKEN` in the environment. Run the statistical baselines for GPT-4o-mini
responses with the paper's scoring and reference models:

```bash
SCORING_MODEL=meta-llama/Llama-3.1-8B-Instruct \
REFERENCE_MODEL=meta-llama/Llama-3.1-8B \
  bash init.sh bash scripts/statistical/run_baseline_statistical_gpt_4o_mini.sh
```

The command evaluates likelihood, log-rank, entropy, LRR, Fast-DetectGPT,
DNA-DetectLLM, and Lastde++. Results are written under
`results/baseline_statistical/`.

`train_bert_detector_baseline.py` and `train_bert_author_attribute.py` train the
embedding-based detection and attribution baselines. List their model, data,
and optimization arguments with:

```bash
bash init.sh python train_bert_detector_baseline.py --help
bash init.sh python train_bert_author_attribute.py --help
```

The corresponding `eval_bert_*` scripts evaluate saved models.
