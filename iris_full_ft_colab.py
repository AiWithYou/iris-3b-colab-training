"""Full Iris fine-tuning for one A100, sharing data/checkpoint primitives."""
from __future__ import annotations

import gc
import importlib.metadata
import math
import random
import shutil
from dataclasses import asdict
from pathlib import Path

import torch

import iris_lora_colab as core

BITSANDBYTES_VERSION = "0.49.2"


class BucketBatchOrder:
    """Deterministic full batches; cover every image, then pad only bucket tails."""

    def __init__(self, records, batch_size, seed):
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("micro_batch_sizeは1以上の整数にしてください。")
        groups = {}
        for index, record in enumerate(records):
            groups.setdefault((record["height"], record["width"]), []).append(index)
        self.groups = [groups[key] for key in sorted(groups)]
        self.batch_size, self.seed = batch_size, seed
        self.epoch_size = sum(math.ceil(len(group) / batch_size) * batch_size for group in self.groups)
        self.padding = self.epoch_size - len(records)

    def order(self, epoch):
        rng = random.Random(self.seed + epoch)
        batches = []
        for group in self.groups:
            indices = list(group)
            rng.shuffle(indices)
            missing = (-len(indices)) % self.batch_size
            indices.extend(indices[i % len(group)] for i in range(missing))
            batches.extend(indices[offset:offset + self.batch_size]
                           for offset in range(0, len(indices), self.batch_size))
        rng.shuffle(batches)
        return [index for batch in batches for index in batch]


def load_full_models(settings, cache_root, model_dir=None, text_dir=None, device="cuda"):
    from huggingface_hub import snapshot_download
    from omegaconf import OmegaConf
    from safetensors.torch import load_file
    from iris3b.config import inference_config
    from iris3b.models import IrisDiT
    from iris3b.text.qwen3_vl import Qwen3VLTextEncoder

    shift = core.flow_shift_for_resolution(settings["resolution"])
    core.seed_all(settings["seed"])
    cache_root = Path(cache_root)
    if model_dir is None:
        model_dir = snapshot_download(core.BASE_REPO, revision=core.BASE_REVISION,
            allow_patterns=["config.yaml", "model.safetensors"], local_dir=str(cache_root / "iris-base"))
    if text_dir is None:
        text_dir = snapshot_download(core.TEXT_REPO, revision=core.TEXT_REVISION,
            allow_patterns=["*.json", "*.safetensors", "merges.txt", "vocab.json", "tokenizer*"],
            local_dir=str(cache_root / "qwen-text"))
    cfg = inference_config(OmegaConf.to_container(OmegaConf.load(Path(model_dir) / "config.yaml")))
    cfg.flow.shift, cfg.flow.shift_law = shift, "none"
    cfg.text_encoder.pretrained = str(text_dir)
    cfg.text_encoder.null_embed_dir = str(cache_root / "null-text")
    cfg.text_encoder.compile = False
    weights = load_file(str(Path(model_dir) / "model.safetensors"), device="cpu")
    with torch.device("meta"):
        model = IrisDiT(cfg.model)
    model.load_state_dict(weights, strict=True, assign=True)
    del weights
    model.to(device=device, dtype=torch.float32).requires_grad_(True)
    model.activation_checkpointing = "full"
    # Construct the frozen decoder on CPU. The trainer moves it to CUDA only
    # before the DiT backward pass, with all previous gradients cleared.
    encoder = Qwen3VLTextEncoder(cfg.text_encoder, device="cpu")
    cfg.text_encoder.pretrained = core.TEXT_REPO
    targets = [name for name, _ in model.named_parameters()]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Iris本体: {sum(p.numel() for p in model.parameters()):,}パラメータを全て学習 / FP32保持・BF16計算")
    return model, encoder, cfg, targets


def storage_and_memory_estimate(model):
    """Tensor-size estimate, excluding activations, CUDA and transient casts."""
    weights = sum(t.numel() * t.element_size() for t in model.state_dict().values())
    gradients = sum(p.numel() * p.element_size() for p in model.parameters())
    optimizer = 0
    for parameter in model.parameters():
        n = parameter.numel()
        optimizer += n * 8 if n < 4096 else n * 2 + math.ceil(n / 256) * 8 + 2048
    checkpoint = weights + optimizer + 64 * 1024**2
    export = sum(t.numel() * (2 if t.is_floating_point() else t.element_size())
                 for t in model.state_dict().values())
    return dict(weight_gib=weights / 1024**3, gradient_gib=gradients / 1024**3,
        optimizer_estimated_gib=optimizer / 1024**3,
        persistent_training_tensor_gib=(weights + gradients + optimizer) / 1024**3,
        checkpoint_estimated_bytes=checkpoint, checkpoint_estimated_gib=checkpoint / 1024**3,
        bf16_export_estimated_gib=export / 1024**3,
        drive_peak_estimated_gib=(3 * checkpoint + export) / 1024**3,
        activations_included=False, measured_on_a100=False)


class FullFTTrainer(core.LoRATrainer):
    checkpoint_tensor_key = "model"
    gradient_clip = 0.5

    def __init__(self, model, encoder, cfg, targets, dataset, settings, run_dir, scratch, device="cuda"):
        settings = dict(settings)
        settings.setdefault("micro_batch_size", 1)
        self.micro_batch_size = settings["micro_batch_size"]
        self.bucket_order = BucketBatchOrder(dataset.records, self.micro_batch_size, settings["seed"])
        super().__init__(model, encoder, cfg, targets, dataset, settings, run_dir, scratch, device)

    def index_at(self, cursor):
        if self.micro_batch_size == 1:
            return super().index_at(cursor)
        epoch, offset = divmod(cursor, self.bucket_order.epoch_size)
        if epoch != self._epoch:
            self._order = self.bucket_order.order(epoch)
            self._epoch = epoch
        return self._order[offset]

    def _validate_trainable_model(self, parameters):
        if (not parameters or core.lora_modules(self.model) or
            any(not p.requires_grad or p.dtype != torch.float32 for p in self.model.parameters())):
            raise ValueError("全体FTはLoRAなし・Iris全パラメータをFP32で学習する構成です。")

    def _create_optimizer(self, parameters):
        if self.device.type == "cuda":
            from bitsandbytes.optim import AdamW8bit
            version = importlib.metadata.version("bitsandbytes")
            if version != BITSANDBYTES_VERSION:
                raise RuntimeError(f"bitsandbytes=={BITSANDBYTES_VERSION}で環境を準備してください。")
            self.optimizer_identity = f"bitsandbytes.AdamW8bit/{version}"
            return AdamW8bit(parameters, lr=self.settings["learning_rate"], betas=(0.9, 0.95),
                             weight_decay=0, min_8bit_size=4096)
        # Only the small CPU behavioral tests use this reference optimizer.
        # The Colab notebook requires a real A100 before reaching this code.
        self.optimizer_identity = "torch.AdamW/cpu-reference-tests"
        return torch.optim.AdamW(parameters, lr=self.settings["learning_rate"],
                                betas=(0.9, 0.95), weight_decay=0)

    def _build_recipe(self, targets):
        keys = ("seed", "learning_rate", "warmup_steps", "grad_accum", "text_dropout")
        recipe = dict(schema=core.SCHEMA_VERSION, training_mode="full-iris-fp32-v1",
            source_revision=core.SOURCE_REVISION, base_repo=core.BASE_REPO,
            base_revision=core.BASE_REVISION, text_repo=core.TEXT_REPO, text_revision=core.TEXT_REVISION,
            dataset_fingerprint=self.dataset.fingerprint, data=core.data_recipe(self.settings),
            training={key: self.settings[key] for key in keys}, targets=list(targets),
            model=asdict(self.cfg.model), flow=asdict(self.cfg.flow), base_dtype="torch.float32",
            compute_dtype="torch.bfloat16" if self.device.type == "cuda" else "torch.float32",
            optimizer=self.optimizer_identity, gradient_clip=self.gradient_clip, ema=False,
            conditioning="qwen-step-batch-cpu-offload-v1")
        if self.micro_batch_size > 1:
            recipe["training"]["micro_batch_size"] = self.micro_batch_size
            recipe["sampler"] = dict(version="seeded-bucket-tail-padding-v1",
                epoch_size=self.bucket_order.epoch_size, padding_per_epoch=self.bucket_order.padding)
        return recipe

    def _autocast(self):
        # Keeping every cast FP32 weight cached adds another ~6 GB. Checkpoint
        # recomputation recasts the active block instead, while updates stay FP32.
        return torch.autocast(self.device.type, dtype=torch.bfloat16,
                              enabled=self.device.type == "cuda", cache_enabled=False)

    def _build_null(self):
        from iris3b.text.base import TextEncoding
        self._conditionings = {}
        try:
            self.encoder.to(self.device)
            with torch.no_grad():
                encoding = self.encoder.null()
                return TextEncoding(encoding.embeddings.detach().cpu(), encoding.mask.detach().cpu())
        finally:
            self.encoder.to("cpu")
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def _prepare_step(self, count=None):
        from iris3b.text.base import TextEncoding
        if any(p.grad is not None for p in self.model.parameters()):
            raise RuntimeError("QwenをGPUへ移す前に前回の勾配を解放してください。")
        count = self.settings["grad_accum"] * self.micro_batch_size if count is None else count
        self._conditionings.clear()
        captions = [self.dataset.records[self.index_at(self.cursor + offset)]["prompt"]
                    for offset in range(count)]
        try:
            self.encoder.to(self.device)
            with torch.no_grad():
                encoding = self.encoder.encode(captions)
                drops = torch.rand(count) < self.settings["text_dropout"]
                for offset in range(count):
                    if bool(drops[offset]):
                        embedding, mask = self.null.embeddings, self.null.mask
                    else:
                        embedding = encoding.embeddings[offset:offset + 1]
                        mask = encoding.mask[offset:offset + 1]
                    self._conditionings[self.cursor + offset] = TextEncoding(
                        embedding.detach().to(self.device), mask.detach().to(self.device))
        finally:
            # Qwen is absent from CUDA during *all* accumulation microsteps.
            self.encoder.to("cpu")
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def _clear_step(self):
        self._conditionings.clear()
        self.encoder.to("cpu")

    def micro_loss(self, cursor):
        cursors = range(cursor, cursor + self.micro_batch_size)
        if any(position not in self._conditionings for position in cursors):
            raise RuntimeError("optimizerステップ前にテキスト条件を準備してください。")
        images = [self.dataset[self.index_at(position)][0] for position in cursors]
        if len({tuple(image.shape) for image in images}) != 1:
            raise RuntimeError("同じ縦横比バケットの画像をバッチにまとめてください。")
        image = torch.stack(images).to(self.device, dtype=torch.float32)
        embedding = torch.cat([self._conditionings[position].embeddings for position in cursors])
        mask = torch.cat([self._conditionings[position].mask for position in cursors])
        with self._autocast():
            return self.flow.training_loss(self.model, image, embedding,
                                           model_kwargs={"y_mask": mask}).loss

    def sanity_check(self):
        state = core.rng_state()
        self.optimizer.zero_grad(set_to_none=True)
        try:
            self._prepare_step(count=self.micro_batch_size)
            super().sanity_check()
        finally:
            self._clear_step()
            core.restore_rng(state)

    def _checkpoint_weights(self):
        return core.cpu_tree(self.model.state_dict())

    def _restore_weights(self, state):
        expected = self.model.state_dict()
        if (state.keys() != expected.keys() or
            any(state[name].shape != tensor.shape or state[name].dtype != tensor.dtype
                for name, tensor in expected.items())):
            raise ValueError("全体FTの保存モデルの名前・形状・dtypeが一致しません。")
        self.model.load_state_dict(state, strict=True)

    def _immutable_setting_keys(self):
        return ("seed", "learning_rate", "warmup_steps", "grad_accum", "micro_batch_size", "text_dropout",
                "sample_count", "resolution", "aspect_buckets", "min_aesthetic_tier", "sfw_only")

    def update_runtime_settings(self, settings):
        settings = dict(settings)
        settings.setdefault("micro_batch_size", 1)
        super().update_runtime_settings(settings)

    def save(self):
        if self.invalid_optimizer_state:
            return super().save()  # same invalid-state guard as LoRA
        if self.step == 0:
            print("学習更新が未完了のため全体FTチェックポイントは保存しません。", flush=True)
            return None
        if self.last_saved_step != self.step:
            needed = storage_and_memory_estimate(self.model)["checkpoint_estimated_bytes"]
            if shutil.disk_usage(self.scratch).free < needed:
                raise OSError("全体FTの途中保存用にColab側のディスク空きを確保してください。")
        return super().save()

    @torch.no_grad()
    def sample(self, prompts, steps=50, seed=42, compare_base=False):
        if compare_base:
            raise ValueError("全体FTのベース比較には、学習前のstep 0作例を使ってください。")
        try:
            self.encoder.to(self.device)
            return super().sample(prompts, steps=steps, seed=seed, compare_base=False)
        finally:
            self.encoder.to("cpu")
            if self.device.type == "cuda":
                torch.cuda.empty_cache()


def export_full_model(trainer, output_dir):
    """Export complete BF16 official Iris weights; keep FP32 resume files intact."""
    from omegaconf import OmegaConf
    from safetensors.torch import save_file
    if trainer.step == 0 or trainer.invalid_optimizer_state:
        raise RuntimeError("正常な学習済みモデルを読み込んでから書き出してください。")
    output_dir = Path(output_dir) / f"step-{trainer.step:08d}"
    staging = trainer.scratch / "full-export"
    staging.mkdir(parents=True, exist_ok=True)
    local_weights, local_config = staging / "model.safetensors", staging / "config.yaml"
    try:
        weights = {name: tensor.detach().cpu().to(torch.bfloat16 if tensor.is_floating_point() else tensor.dtype)
                   .contiguous().clone() for name, tensor in trainer.model.state_dict().items()}
        save_file(weights, str(local_weights))
        del weights
        text = asdict(trainer.cfg.text_encoder)
        text["pretrained"], text["null_embed_dir"] = core.TEXT_REPO, "output/pretrained_models"
        OmegaConf.save(OmegaConf.create(dict(model=asdict(trainer.cfg.model), text_encoder=text,
                                            flow=asdict(trainer.cfg.flow))), local_config)
        weights_hash = core.verified_copy(local_weights, output_dir / local_weights.name)
        config_hash = core.verified_copy(local_config, output_dir / local_config.name)
        core.atomic_json(output_dir / "export_manifest.json", dict(format="iris3b-full-bf16-v1",
            optimizer_step=trainer.step, signature=trainer.signature, recipe=trainer.recipe,
            files={"model.safetensors": weights_hash, "config.yaml": config_hash}))
    finally:
        local_weights.unlink(missing_ok=True)
        local_config.unlink(missing_ok=True)
    print(f"全体FTモデルを公式Iris形式で保存しました: {output_dir}")
    return output_dir
