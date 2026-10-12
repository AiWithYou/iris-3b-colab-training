# Iris-3B Colab Training

[日本語](README.md) | **English**

Self-contained Colab notebooks for Iris-3B LoRA and full fine-tuning, with configurable datasets and resumable Google Drive checkpoints.

These notebooks use the [official Iris-3B implementation](https://github.com/speridlabs/iris-3b) for the model architecture, Qwen text encoder, Rectified Flow, and sampler. This repository provides LoRA adapters, single-GPU full fine-tuning, dataset selection, checkpoint saving and resuming, and continuation on new data.

## Open in Colab

| Purpose | Default configuration | Notebook |
| --- | --- | --- |
| Full fine-tuning | 1024px, micro-batch 14 × accumulation 2, 300 optimizer updates | [Open in Colab](https://colab.research.google.com/github/AiWithYou/iris-3b-colab-training/blob/main/Iris3B_FullFT_A100.ipynb) |
| Continue full fine-tuning on new data | Retain optimizer, scheduler, and RNG state | [Open in Colab](https://colab.research.google.com/github/AiWithYou/iris-3b-colab-training/blob/main/Iris3B_FullFT_Continuation.ipynb) |
| LoRA | 1024px, rank 32, batch 1 × accumulation 8 | [Open in Colab](https://colab.research.google.com/github/AiWithYou/iris-3b-colab-training/blob/main/Iris3B_LoRA_A100.ipynb) |
| LoRA at 512px | 512px, rank 16, batch 1 × accumulation 8 | [Open in Colab](https://colab.research.google.com/github/AiWithYou/iris-3b-colab-training/blob/main/Iris3B_LoRA_A100_512.ipynb) |

1. Select an A100 GPU. The default full fine-tuning micro-batch of 14 requires an A100 80GB and a high-RAM runtime.
2. Enter the dataset settings below and your own `RUN_NAME` in the training configuration cell.
3. Run the cells in order. Inspect the selected images, first optimizer update, and checkpoint before continuing training.
4. In a new runtime, run the cells again with the same settings. The latest valid checkpoint is restored automatically.

See the [full fine-tuning guide](README_FULL_FT_EN.md) and the [continuation guide](README_CONTINUATION_EN.md) for details.

## Choose your training data

The notebooks read Parquet files from a user-selected Hugging Face dataset repository. The repository and revision fields are blank by default. Direct loading from image folders or CSV files is not supported.

| Setting | What to enter |
| --- | --- |
| `DATASET_REPO` | Your dataset's `owner/repository` |
| `DATASET_REVISION` | The dataset's full 40-character commit SHA. Mutable refs such as `main` are not accepted |
| `PARQUET_GLOB` | File pattern to read. The default `*.parquet` also matches files in subdirectories |
| `ID_COLUMN` | Column containing a unique image ID. Default: `id` |
| `IMAGE_COLUMN` | Image column. Default: `image`. Supports image bytes or an HF Image struct with `bytes` and `path` fields |
| `PROMPT_COLUMN` | Column containing a caption or tag string. Default: `prompt` |
| `QUALITY_COLUMN` | Numeric quality-tier column. Leave blank if unused |
| `MIN_AESTHETIC_TIER` | Minimum quality tier. Use 0 when there is no quality column |

Images must be embedded in the Parquet files. Image URLs and path-only entries with empty `bytes` are not loaded. Supported formats include PNG, JPEG, AVIF, and other formats that Pillow can decode.

Selection excludes empty captions, duplicate IDs, and unreadable images, then collects the requested number of images. Shard order and the rows within each 64-row reading batch are shuffled with the configured seed. This is not strictly uniform sampling across the entire dataset.

`SFW_ONLY` filters by rating tags in the caption. Leave it False for natural-language captions or other data without rating tags. Check your dataset's usage terms and image content. For private or gated datasets, authenticate in Colab with `huggingface_hub.login()`.

## Checkpoints and resuming

Runs are saved under `MyDrive/Iris3B_LoRA/<RUN_NAME>/` or `MyDrive/Iris3B_FullFT/<RUN_NAME>/`. The selected dataset is also saved so that the same images and order can be restored.

Checkpoints contain trainable weights, optimizer and scheduler state, the optimizer update count, data position, and Python, NumPy, and PyTorch RNG state. Files are checked with SHA-256. If the latest checkpoint is corrupt, an earlier valid generation is used. If all generations are corrupt, execution stops.

Resuming the same run with a different dataset, revision, column mapping, resolution, learning rate, or other fixed training settings is rejected. You can change the target update count, session time limit, and save intervals. Do not run the same run concurrently in multiple runtimes.

A full fine-tuning checkpoint is approximately 18GB per generation, and a BF16 inference model is approximately 6GB. Around 200GB of free Drive storage is recommended for an initial run. Existing checkpoints, the dataset, and Drive Trash also consume storage. The session time limit is not a spending cap.

LoRA exports use this repository's Iris-specific format and are not compatible with standard SDXL or FLUX LoRA loaders. Merging produces the official `model.safetensors` and `config.yaml` format.

## Source and validation

| File | Purpose |
| --- | --- |
| `iris_lora_colab.py` | LoRA, dataset selection, checkpoint saving and resuming, and exports |
| `iris_full_ft_colab.py` | Updates to all Iris parameters, micro-batches, AdamW8bit, and official-format exports |
| `iris_full_ft_phase_colab.py` | Retain training state while continuing on new data |
| `iris_unseen_colab.py` | Build exclusion ledgers from previous image IDs and SHA-256 hashes |
| `iris_preview_colab.py` | Save previews once at a selected optimizer step and restore training state |
| `build_*.py` | Generate notebooks with embedded source code |
| `tests/` | CPU tests using a small instance of the actual Iris architecture and synthetic images |

The official source, base model, and Qwen revisions are pinned in the code. CPU tests cover the public version's dataset configuration and Parquet column mapping. End-to-end A100 execution with an arbitrary dataset and resulting image quality have not been verified.

Regenerate the notebooks with Python 3.11 or later:

```sh
python build_full_notebook.py
python build_continuation_notebook.py
```

To run the CPU tests, install the dependencies and fetch the pinned Iris source. These tests do not download model weights or training images.

```sh
python -m pip install "torch>=2.7.1,<2.14" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-dev.txt
git clone https://github.com/speridlabs/iris-3b.git vendor/iris-3b
git -C vendor/iris-3b checkout a8d15239dea469aba042cfa56ca3bb4e450d5ebc
```

Linux/macOS:

```sh
PYTHONPATH=vendor/iris-3b/src python -m pytest -q tests
```

PowerShell:

```powershell
$env:PYTHONPATH = (Resolve-Path vendor/iris-3b/src).Path
python -m pytest -q tests
```

## License

The code is licensed under the [Apache License 2.0](LICENSE). Official Iris copyright notices are retained in [NOTICE](NOTICE) and [third_party/iris-NOTICE](third_party/iris-NOTICE). External models, dependencies, and user-selected datasets retain their respective licenses.
