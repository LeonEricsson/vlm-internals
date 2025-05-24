# inspecting vlm internals

this repository contains small-scale tooling for studying language models—currently focused only on vision-language models. the repo is intentionally messy and research-oriented. it’s not meant to grow into a large project, but rather stay quick and dirty, scriptable, and easy to understand.

my personal focus is on vlm reasoning, specifically spatial reasoning. i'm interested in understanding how and why vlms may struggle with spatial reasoning, and what effect performing such reasoning through text might have.

extending the repo to support other domains, models, etc., should be straightforward. the code is intentionally lightweight to enable fast experimentation.

## Repository Overview

- **`data/`** – everything dataset related, downloading, preprocessing etc.
- **`model_utils/`** – wrappers and collators for specific VLMs (currently Qwen VL 2.5). Wrappers implement a forward pass wrapper for extracting internals, collators prepare batches for model specific input.
- **`analysis/`** - helper functions for analyzing and visualizing extracted model internals.
- **`extract.py`** – run a model on a dataset and save model internals to compressed `.npz` files.
- **`analyze.py`** – entry point for utilities to inspect the extracted tensors,
- **`evaluate.py`** – basic accuracy evaluation of a model on the VSR dataset.

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
