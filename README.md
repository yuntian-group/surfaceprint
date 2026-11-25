# SurfacePrint: Typographic Fingerprints of LLMs

Code for **SurfacePrint: Typographic Fingerprints of LLMs**, EACL 2027.

Wentao Zhang, Sajad Ebrahimi, Nima Jamali, Bardia Shirsalimian, Kelly McConvey,
Jalehsadat Mahdavimoghaddam, Maksym Taranukhin, Maura R. Grossman, Vered Shwartz,
Ebrahim Bagheri, and Yuntian Deng.

SurfacePrint detects AI-generated text using 97 features of punctuation,
whitespace, Markdown, LaTeX, and code formatting. Feature extraction and random
forest training run on CPU. The paper reports a mean per-generator AUROC of
0.983 across 17 LLMs and four text domains.

## Installation

Use Python 3.11 or newer. Run the commands below from the repository root.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The [GPU baseline instructions](docs/experiments.md#gpu-baselines) use a separate
Apptainer environment.

## Data

Download [merged_dataset.tar.zst](https://drive.google.com/file/d/1Olau3eFQmb0nt5YuD_v4tDuLipxdt07p/view)
(163 MiB), place it in `dataset/`, and extract it with `zstd` installed:

```bash
mkdir -p dataset
zstd -dc dataset/merged_dataset.tar.zst | tar -xf - -C dataset
```

The extracted Hugging Face dataset has `train`, `val`, and `test` splits at
`dataset/merged_dataset/`. Each row contains a prompt, response, generator, and
source dataset. See [the data format and sources](docs/data.md) for column
definitions and the exact model identifiers used by the scripts.

## Train and evaluate

Train one random forest on human responses and responses from all 17
instruction-tuned models, then evaluate it on the test split:

```bash
bash scripts/train_unified_raw.sh
```

The command writes:

- `models/surfaceprint_raw/surfaceprint_raw.pkl`: the trained classifier.
- `models/surfaceprint_raw/surfaceprint_raw.metrics.json`: training and test metrics, per-generator AUROC/AUPRC, and preprocessing settings.

`mean_per_model_test_auroc` is the average of the per-generator AUROC values.
Each generator is evaluated against the human test responses.

Set the data and output directories without editing the script:

```bash
DATASET_PATH=/path/to/merged_dataset OUTPUT_DIR=models/surfaceprint \
  bash scripts/train_unified_raw.sh
```

To train after removing ASCII spaces, tabs, carriage returns, and newlines:

```bash
bash scripts/train_unified_ws.sh
```

The raw variant strips leading and trailing whitespace and preserves formatting
inside the response. The two preprocessing modes are defined in
[`normalize_text`](train_merged_ascii_detector.py).

## Detect AI-generated text

After training, score a UTF-8 text file:

```bash
python predict.py \
  --model models/surfaceprint_raw/surfaceprint_raw.pkl \
  --file article.txt
```

The command reads the adjacent `.metrics.json` file to use the same features
and preprocessing as training. It prints one JSON object per input file, for example:

```json
{"file": "article.txt", "ai_score": 0.87}
```

`ai_score` is the classifier's probability for the
AI class, between 0 and 1. Higher scores indicate AI-generated text. Pass
several paths after `--file` to score several documents, or use `--text` for a
single string.

## Experiments

[Experiment commands](docs/experiments.md) cover per-model detection, shallow
decision trees, model attribution, transfer to unseen models, and GPU baselines.

The feature definitions are in
[`detectors/ascii_detector.py`](detectors/ascii_detector.py). Their order is
listed in `FEATURE_NAMES`; `FEATURE_GROUPS` maps each feature to punctuation,
Markdown, math, code, or whitespace.

## Citation

```bibtex
@inproceedings{zhang2027surfaceprint,
  title = {SurfacePrint: Typographic Fingerprints of LLMs},
  author = {Zhang, Wentao and Ebrahimi, Sajad and Jamali, Nima and
            Shirsalimian, Bardia and McConvey, Kelly and
            Mahdavimoghaddam, Jalehsadat and Taranukhin, Maksym and
            Grossman, Maura R. and Shwartz, Vered and Bagheri, Ebrahim and
            Deng, Yuntian},
  booktitle = {Proceedings of the 20th Conference of the European Chapter
               of the Association for Computational Linguistics},
  year = {2027}
}
```
