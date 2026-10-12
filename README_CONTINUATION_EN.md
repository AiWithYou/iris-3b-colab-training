# Continue Full Fine-Tuning on New Data

[日本語](README_CONTINUATION.md) | **English**

[Open in Colab](https://colab.research.google.com/github/AiWithYou/iris-3b-colab-training/blob/main/Iris3B_FullFT_Continuation.ipynb)

Continue from a full fine-tuning checkpoint created with this public version, using a new image pool. Stop the parent run and retain its checkpoint files. Resuming the continuation also requires the identity of the parent checkpoint used when the phase started.

1. Set `PARENT_RUN_NAME` and choose a different name for `RUN_NAME`.
2. Enter the new dataset repository, pinned commit SHA, column names, image count, and quality conditions.
3. Match the parent's seed, resolution, aspect-ratio processing, micro-batch, gradient accumulation, warmup, text dropout, and other fixed training settings.
4. Set `MAX_STEPS` to the cumulative target. For example, a target of 600 after a parent run of 300 updates adds 300 updates.
5. To change the learning rate, enter the parent's configured value in `PARENT_LEARNING_RATE` and the new value in `LEARNING_RATE`. Optimizer moments and scheduler progress are retained.

The parent's FP32 weights, optimizer, scheduler, RNG state, and cumulative data cursor are restored. The new dataset's local cursor starts at zero. You can change the dataset repository, revision, column mapping, image count, and quality conditions. Mismatched resolution or preprocessing causes execution to stop.

To exclude previously used images, set `EXCLUSION_MANIFESTS_JSON` to a list such as `["datasets/<dataset-key>/manifest.json"]`. Find the actual `<dataset-key>` in the parent's saved dataset folder. You can include manifests from multiple earlier phases.

Each previous manifest's fingerprint is verified before image IDs and SHA-256 hashes of the original image bytes are checked. Image IDs must be unique across the datasets you combine. The exclusion ledger and new dataset are bound to the run's checkpoint identity and verified when resuming.

Use `iris_preview_colab.preview_once(..., target_step=...)` to save previews once at a chosen optimizer update. Call it after saving the checkpoint at that step. It restores model, optimizer, scheduler, RNG state, and model mode before training continues. The notebook does not automatically generate additional intermediate previews.

After training, export the official-format BF16 model, synchronize Drive, and disconnect the GPU. End-to-end A100 execution of the public version has not been verified.
