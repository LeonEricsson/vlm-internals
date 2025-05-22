# Vision-Language Reasoning Research

This repository contains small-scale tooling for studying **vision-language models (VLMs)**. It focuses on evaluating reasoning ability and analysing model internals such as attention patterns. The code is intentionally lightweight to support quick experiments.

## Repository Overview

- **`data/`** – Dataset helpers for the VSR benchmark. `get_dataset()` can be extended to load your own tasks.
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

## Extending the Repo

1. **New datasets** – Implement a dataset class in `data/` and update `get_dataset()` to dispatch to it.
2. **New models** – Create a wrapper in `model_utils/` and register it in `_WRAPPERS` within `model_utils/__init__.py`.
3. **Custom analyses** – `analysis/` contains helper functions for loading extractions and plotting. Feel free to build additional scripts on top.

Pull requests are welcome for small improvements, but this repo is primarily used for research prototypes.

## License

This project is licensed under the Apache 2.0 License. See `LICENSE` for details.
