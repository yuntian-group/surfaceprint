# Data

The [Hugging Face dataset](https://huggingface.co/datasets/yuntian-group/surfaceprint) contains 334,976 rows of human writing and model generations from four domains.

```python
from datasets import load_dataset

data = load_dataset("yuntian-group/surfaceprint")
for split, rows in data.items():
    print(split, len(rows), rows.column_names)
data.save_to_disk("dataset/merged_dataset")
```

## Fields

Each row has five string fields:

- `prompt`: the question or writing prompt.
- `response`: the human answer or story, or the model's generated response.
- `data_name`: the source-dataset identifier, such as `euclaise/writingprompts`.
- `source`: `human` or the model identifier used by the experiment scripts.
- `dataset`: one of `writingprompts`, `reddit_eli5`, `stackmathqa`, or `stackprogram`.

The detector uses `response` by default. For binary detection, `source == "human"` receives label 0 and model generations receive label 1.

## Splits

The dataset has 191,372 rows in `train`, 47,841 in `val`, and 95,763 in `test`. Use `train` to fit detectors, `val` to select settings, and `test` to report results. The unified detector scripts train on `train` and evaluate on `test`.

The paper's sampling design uses 2,000 training prompts, 500 validation prompts, and 1,000 test prompts from each domain. Prompt counts differ from row counts: a prompt has human writing and generations from multiple models. [Per-domain and per-model counts](https://huggingface.co/datasets/yuntian-group/surfaceprint/blob/main/statistics.json) are included with the dataset.

## Domains and original datasets

- `writingprompts`: Reddit stories paired with writing prompts, from [WritingPrompts](https://huggingface.co/datasets/euclaise/writingprompts) ([Fan et al., 2018](https://aclanthology.org/P18-1082/)). The dataset card lists MIT.
- `reddit_eli5`: human answers to Reddit ELI5 questions, from [HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3). Its card lists CC BY-SA 4.0 and retains stricter source-dataset terms.
- `stackmathqa`: mathematics questions and answers from [StackMathQA](https://huggingface.co/datasets/math-ai/StackMathQA). Its card lists CC BY 4.0.
- `stackprogram`: programming questions and answers from [Hugging Face H4 Stack Exchange Preferences](https://huggingface.co/datasets/HuggingFaceH4/stack-exchange-preferences), restricted to programming topics and posts before November 2022. Its card lists CC BY-SA 4.0.

These licenses apply to the original questions and human responses. [Programming post and author attribution](https://huggingface.co/datasets/yuntian-group/surfaceprint/blob/main/attribution/stackprogram.jsonl) is stored separately from the five text fields.

## Model identifiers

The unified detector scripts select these 17 `source` values:

```text
gpt-3.5-turbo
gpt-4o-mini
gpt-4.1-mini
gpt-5.4-mini
anthropic/claude-3-haiku
anthropic/claude-3.5-haiku
anthropic/claude-sonnet-4
claude-sonnet-4-6
google/gemini-2.0-flash-001
gemini-2.5-flash
gemini-3.5-flash
facebook/opt-iml-1.3b
/model-weights/Llama-2-7b-chat-hf/
meta-llama/Llama-3.1-8B-Instruct
allenai/Olmo-3-7B-Instruct
/model-weights/Qwen3.5-9B
google/gemma-4-E4B-it
```

The per-model scripts also select six base models:

```text
facebook/opt-1.3b
/model-weights/Llama-2-7b-hf/
meta-llama/Llama-3.1-8B
allenai/Olmo-3-1025-7B
Qwen/Qwen3.5-9B-Base
google/gemma-4-E4B
```

The `/model-weights/...` strings are stored labels, not directories the detector needs to access. They identify Llama 2 7B, Llama 2 7B Chat, and Qwen3.5 9B. Use the exact strings, including trailing slashes, with `--include-source` and `--target-source`.

## Text preprocessing

`--normalize-mode raw` applies Python's `str.strip()`: it removes leading and trailing whitespace and preserves whitespace inside the text.

`--normalize-mode whitespace` deletes ASCII spaces (`U+0020`), tabs (`U+0009`), carriage returns (`U+000D`), and newlines (`U+000A`) throughout the text. Other Unicode whitespace is unchanged.

`--text-field prompt_response` joins `prompt` and `response` with two newlines before applying the selected preprocessing. The default `--text-field response` uses the response alone.

Keep the original strings when copying or converting this dataset. Whitespace, apostrophes, and other Unicode characters are inputs to SurfacePrint's features.
