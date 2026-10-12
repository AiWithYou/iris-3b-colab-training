# Iris-3B Full Fine-Tuning

[日本語](README_FULL_FT.md) | **English**

[Open in Colab](https://colab.research.google.com/github/AiWithYou/iris-3b-colab-training/blob/main/Iris3B_FullFT_A100.ipynb)

All Iris parameters are updated in FP32, while Qwen remains frozen. Training uses AdamW8bit, BF16 autocast, and gradient checkpointing. EMA is not retained. This recipe differs from the official distributed training recipe.

Defaults are aspect-ratio buckets at approximately 1024px resolution, 5,000 images, micro-batch 14 × gradient accumulation 2, effective batch size 28, learning rate 1e-5, 30 warmup updates, and 300 optimizer updates. The resolution-specific flow shift is 3 at 512px and 4 at 1024px.

Micro-batch 14 is intended for an A100 80GB. Memory fit on an A100 40GB has not been verified; start with micro-batch 1 if testing that configuration. Use Colab's high-RAM option because checkpoint saving also needs CPU memory.

Configure the dataset repository, full 40-character commit SHA, and column names. If there is no quality column, leave `QUALITY_COLUMN` blank and set `MIN_AESTHETIC_TIER=0`. See [dataset settings](README_EN.md#choose-your-training-data).

The first training cell performs one optimizer update and saves the full weights, optimizer state, and other training state to Drive. This counts as the first update of the actual run. Check memory usage, loss, gradients, and the save result before proceeding.

A resumable checkpoint is approximately 18GB per generation, and the BF16 model is approximately 6GB. Two existing generations and a new checkpoint can coexist during saving. The dataset and Drive Trash also consume storage. Around 200GB of free Drive storage is recommended for an initial run. A full fine-tuning checkpoint does not fit in the free 15GB Drive allowance.

If Colab disconnects, run the cells again with the same `RUN_NAME` and settings. Checkpoint SHA-256 hashes are verified before restoring a valid generation. Increasing the target update count preserves the existing scheduler state.

Exports contain `export/step-XXXXXXXX/model.safetensors` and `config.yaml`. Keep both files in the same folder and pass that folder to the official Iris `--checkpoint` option. To resume training, use the full weights and optimizer checkpoint under `checkpoints`, rather than the BF16 inference export.

Finally, run the Drive synchronization cell and disconnect the GPU runtime. GitHub Actions runs CPU tests covering this public version's settings and dataset conversion. End-to-end A100 execution of the public version has not been verified.
