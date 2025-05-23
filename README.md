# inspecting vlm internals

This repository contains small-scale tooling for studying **vision-language models (VLMs)**. The focus is on mechanistic interpretability techniques. My personal agenda is an interest in VLM reasoning, spatial reasoning to be exact. I'd like to understand how and why VLMs may struggle with spatial reasoning, and the effect of performing spatial reasoning through text. However, extending to other domains, models, etc should be straightforward. The code is intentionally lightweight to support quick experiments.

## Repository Overview

- **`data/`** – Dataset helpers for downloading and preprocessing, currently only supports VSR.
- **`model_utils/`** – Wrappers and collators for specific VLMs (currently Qwen 2.5). Add new wrappers here and register them in `get_wrapper_collator()`.
- **`extract.py`** – Run a model on a dataset and save attention maps, hidden states and prediction metadata to compressed `.npz` files.
- **`analyze.py`** – Utilities to inspect the saved tensors, including attention map visualisation and simple statistics.
- **`evaluate.py`** – Basic accuracy evaluation of a model on the VSR dataset.

## Setup

The project requires **Python 3.12+**. Dependencies are defined in `pyproject.toml`. You can install them with `pip` or `uv`:

```bash
pip install -e .
# or
uv pip install -e .
```

## Example Usage

Evaluate a model on VSR:

```bash
python evaluate.py --model Qwen/Qwen2.5-VL-3B-Instruct
```

Extract attention weights and activations:

```bash
python extract.py --model Qwen/Qwen2.5-VL-3B-Instruct --output_dir ./extractions
```

Visualise attention maps:

```bash
python analyze.py --input_dir ./extractions/Qwen_Qwen2.5-VL-3B-Instruct_VSR/ --sample_idx 0
```

## License

This project is licensed under the Apache 2.0 License. See `LICENSE` for details.
