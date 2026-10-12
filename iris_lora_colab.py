"""Iris-3B LoRA training, fixed data selection, and resumable Drive checkpoints.

The notebook embeds this file verbatim. The original Iris architecture, text
encoder, rectified-flow loss, and sampler are imported from the pinned upstream.
"""
from __future__ import annotations

import copy
import errno
import fnmatch
import gc
import hashlib
import io
import json
import math
import os
import pickle
import random
import re
import shutil
import tarfile
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps, features
from torch import nn

SOURCE_REVISION = "a8d15239dea469aba042cfa56ca3bb4e450d5ebc"
BASE_REPO = "speridlabs/iris-3b"
BASE_REVISION = "7445443349bc9abe3c96f01ff793e2098ca012b3"
TEXT_REPO = "Qwen/Qwen3-VL-4B-Instruct"
TEXT_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
DEFAULT_DATA_COLUMNS = {
    "image_id": "id", "image": "image", "prompt": "prompt", "aesthetic_tier": None,
}
SCHEMA_VERSION = 2
IMAGE_NAME = re.compile(r"images/[0-9]{6}\.bin\Z")
CHECKPOINT_NAME = re.compile(r"step-[0-9]{8}-[a-f0-9]{8}\.pt\Z")


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def flush_file(stream):
    stream.flush()
    try:
        os.fsync(stream.fileno())
    except OSError as exc:
        # Drive/FUSE implementations may not expose fsync. Readback checks
        # still apply; real I/O and quota failures must remain errors.
        if exc.errno not in {errno.EINVAL, errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP}:
            raise


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")
    try:
        serialized = json.dumps(value, ensure_ascii=False, indent=2)
        with temp.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
            flush_file(stream)
        if file_hash(temp) != hashlib.sha256(serialized.encode("utf-8")).hexdigest():
            raise OSError("JSON保存のSHA-256が一致しません。")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def verified_copy(source, destination, expected_hash=None):
    """Publish the destination only after a complete copy and checksum readback."""
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    checksum = expected_hash or file_hash(source)
    temp = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".partial")
    try:
        with source.open("rb") as incoming, temp.open("wb") as outgoing:
            shutil.copyfileobj(incoming, outgoing, length=8 * 1024 * 1024)
            flush_file(outgoing)
        if file_hash(temp) != checksum:
            raise OSError("保存ファイルのSHA-256が一致しません。")
        os.replace(temp, destination)
    finally:
        temp.unlink(missing_ok=True)
    return checksum


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    np_state = np.random.get_state()
    return dict(python=random.getstate(), torch=torch.get_rng_state().clone(),
                cuda=[x.clone().cpu() for x in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_available() else [],
                numpy=[np_state[0], np_state[1].tolist(), np_state[2], np_state[3], np_state[4]])


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    np_state = state["numpy"]
    np.random.set_state((np_state[0], np.asarray(np_state[1], dtype=np.uint32),
                         np_state[2], np_state[3], np_state[4]))
    if state["cuda"] and torch.cuda.is_available():
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("保存時とGPU台数が違います。1GPUのランタイムを選んでください。")
        torch.cuda.set_rng_state_all(state["cuda"])


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(item) for item in value)
    return value


class LoRALinear(nn.Module):
    """Frozen Linear + FP32 trainable factors, without mutating shared cores."""
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.in_features, self.out_features = base.in_features, base.out_features
        self.rank, self.alpha = int(rank), float(alpha)
        self.scaling = self.alpha / self.rank
        self.enabled = True
        self.lora_A = nn.Parameter(torch.empty(self.rank, self.in_features,
                                               device=base.weight.device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.rank,
                                               device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        result = self.base(x)
        if self.enabled:
            update = F.linear(F.linear(x.to(self.lora_A.dtype), self.lora_A), self.lora_B)
            result = result + (update * self.scaling).to(result.dtype)
        return result


def lora_modules(model):
    return {name: module for name, module in model.named_modules() if isinstance(module, LoRALinear)}


def inject_lora(model, rank=16, alpha=16):
    if rank <= 0 or alpha <= 0:
        raise ValueError("rankとalphaは正の値にしてください。")
    if lora_modules(model):
        raise ValueError("このモデルには既にLoRAが入っています。読み込みセルから再実行してください。")
    model.requires_grad_(False)
    targets = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not name.startswith(("blocks.", "pixel_blocks.", "y_embedder.", "s_embedder.")):
            continue
        if any(part.startswith("adaln") for part in name.split(".")):
            continue
        if min(module.in_features, module.out_features) < 16:
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, LoRALinear(module, rank, alpha))
        targets.append(name)
    if not targets:
        raise ValueError("LoRAを適用できるLinearが見つかりません。Irisのrevisionを確認してください。")
    return targets


def adapter_state(model):
    return {name: tensor.detach().float().cpu().contiguous().clone()
            for name, tensor in model.named_parameters() if name.endswith((".lora_A", ".lora_B"))}


def load_adapter_state(model, state):
    live = {name: parameter for name, parameter in model.named_parameters()
            if name.endswith((".lora_A", ".lora_B"))}
    if set(live) != set(state) or any(live[k].shape != state[k].shape for k in live):
        raise ValueError("保存LoRAとモデルのLinear名・shapeが一致しません。")
    with torch.no_grad():
        for name, parameter in live.items():
            parameter.copy_(state[name].to(device=parameter.device, dtype=parameter.dtype))


def merged_state_dict(model):
    modules = lora_modules(model)
    replacements = {}
    updates = {}
    for name, module in modules.items():
        replacements[name + ".base.weight"] = name + ".weight"
        if module.base.bias is not None:
            replacements[name + ".base.bias"] = name + ".bias"
        updates[name + ".weight"] = (module.lora_B.detach().float().cpu()
                                      @ module.lora_A.detach().float().cpu()) * module.scaling
    output = {}
    for name, tensor in model.state_dict().items():
        if name.endswith((".lora_A", ".lora_B")):
            continue
        target = replacements.get(name, name)
        base = tensor.detach().cpu()
        if target in updates:
            base = (base.float() + updates[target]).to(base.dtype)
        output[target] = base.contiguous().clone()
    return output


def dataset_config(settings, *, require_source=False):
    """Bind a user-selected dataset, commit, Parquet pattern and column mapping."""
    repo = str(settings.get("dataset_repo", "")).strip()
    revision = str(settings.get("dataset_revision", "")).strip()
    columns = dict(DEFAULT_DATA_COLUMNS)
    overrides = settings.get("dataset_columns", {})
    if not isinstance(overrides, dict) or set(overrides) - set(columns):
        raise ValueError("dataset_columns contains unsupported column keys.")
    columns.update(overrides)
    for key, value in columns.items():
        if key == "aesthetic_tier" and value in (None, ""):
            columns[key] = None
        elif not isinstance(value, str) or not value.strip():
            raise ValueError(f"Set a nonempty dataset column for {key}.")
        else:
            columns[key] = value.strip()
    pattern = settings.get("parquet_glob", "*.parquet")
    if not isinstance(pattern, str) or not pattern.strip():
        raise ValueError("parquet_glob must be a nonempty file pattern.")
    if require_source:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
            raise ValueError("Set DATASET_REPO to your Hugging Face dataset repository.")
        if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            raise ValueError("Set DATASET_REVISION to the dataset's full 40-character commit SHA.")
        if columns["aesthetic_tier"] is None and settings.get("min_aesthetic_tier", 0) > 0:
            raise ValueError("Set QUALITY_COLUMN, or use MIN_AESTHETIC_TIER=0 without a quality column.")
    return dict(repo=repo, revision=revision, columns=columns, parquet_glob=pattern.strip())


def data_recipe(settings):
    recipe = {key: settings[key] for key in ("seed", "resolution", "aspect_buckets",
              "sample_count", "min_aesthetic_tier", "sfw_only")}
    recipe.update(schema=SCHEMA_VERSION, **dataset_config(settings),
                  selection="seeded-shards-shuffled-batches-v1", image_transform="lanczos-center-fit-v1")
    if "exclusion" in settings:
        recipe["exclusion"] = copy.deepcopy(settings["exclusion"])
        recipe["selection"] = "seeded-shards-unseen-images-v1"
    return recipe


def flow_shift_for_resolution(resolution):
    """Use the official 512/1024 training-stage noise schedules."""
    if resolution not in (512, 1024):
        raise ValueError("学習解像度は512または1024を選んでください。")
    return 3.0 if resolution == 512 else 4.0


class PreparedDataset:
    def __init__(self, root, records, fingerprint, drive_dir=None):
        self.root, self.records, self.fingerprint = Path(root), records, fingerprint
        self.drive_dir = Path(drive_dir) if drive_dir else None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        with Image.open(self.root / row["path"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image = ImageOps.fit(image, (row["width"], row["height"]),
                                 method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
            pixels = np.asarray(image, dtype=np.float32).copy()
        return torch.from_numpy(pixels).permute(2, 0, 1).contiguous().div_(127.5).sub_(1), row["prompt"]


def dataset_rows(settings, scratch):
    """Download only the seeded shards consumed, decode bounded Arrow batches."""
    config = dataset_config(settings, require_source=True)
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq
    files = [name for name in HfApi().list_repo_files(config["repo"], repo_type="dataset", revision=config["revision"])
             if name.endswith(".parquet") and fnmatch.fnmatchcase(name, config["parquet_glob"])]
    files.sort()
    rng = random.Random(settings["seed"])
    rng.shuffle(files)
    if not files:
        raise RuntimeError("データセットにParquetシャードがありません。")
    cache = Path(scratch) / "shard-downloads"
    for name in files:
        print(f"データを読み込み中: {name}", flush=True)
        path = Path(hf_hub_download(config["repo"], name, repo_type="dataset", revision=config["revision"],
                                    local_dir=str(cache)))
        parquet = pq.ParquetFile(path, memory_map=True)
        try:
            selected_columns = list(dict.fromkeys(name for name in config["columns"].values() if name))
            missing = set(selected_columns) - set(parquet.schema_arrow.names)
            if missing:
                raise ValueError(f"Missing Parquet columns in {name}: {sorted(missing)}")
            for batch in parquet.iter_batches(batch_size=64, columns=selected_columns):
                rows = batch.to_pylist()
                rng.shuffle(rows)
                for row in rows:
                    mapped = {key: row[column] if column else 0 for key, column in config["columns"].items()}
                    raw = mapped["image"]
                    if isinstance(raw, dict):
                        raw = raw.get("bytes")
                    if not isinstance(raw, (bytes, bytearray, memoryview)):
                        raise ValueError("The image column must contain embedded bytes, or an HF Image bytes struct.")
                    mapped["image"] = bytes(raw)
                    yield mapped
        finally:
            parquet.close()
            # Only this task-owned download is removed; selected images are retained.
            path.unlink(missing_ok=True)


def _is_sfw(prompt):
    tags = {tag.strip().lower() for tag in prompt.split(",")}
    allowed = {"safe", "general", "rating:general", "rating:safe", "rating:g"}
    excluded = {"sensitive", "questionable", "explicit", "rating:s", "rating:q", "rating:e"}
    return bool(tags & allowed) and not bool(tags & excluded)


def _restore_dataset(drive_dir, local_dir, manifest, recipe):
    if manifest.get("recipe") != recipe or len(manifest.get("records", [])) != recipe["sample_count"]:
        raise ValueError("保存データの設定・画像数が一致しません。")
    expected = stable_hash({"recipe": recipe, "records": manifest["records"]})
    if manifest.get("fingerprint") != expected:
        raise RuntimeError("保存データのmanifestが破損しています。")
    archive_name = manifest["archive"]
    if not re.fullmatch(r"dataset-[a-f0-9]{12}\.tar", archive_name):
        raise RuntimeError("保存データのarchive名が不正です。")
    records = manifest["records"]
    if any(not IMAGE_NAME.fullmatch(row["path"]) for row in records):
        raise RuntimeError("保存データの画像パスが不正です。")
    ready_path = local_dir / "ready.json"
    ready = json.loads(ready_path.read_text()) if ready_path.is_file() else {}
    if ready.get("fingerprint") != expected or not all((local_dir / row["path"]).is_file() for row in records):
        local_dir.mkdir(parents=True, exist_ok=True)
        archive = local_dir / "restore.tar"
        try:
            verified_copy(drive_dir / archive_name, archive, manifest["archive_sha256"])
        except (OSError, FileNotFoundError) as exc:
            raise RuntimeError("保存データのアーカイブが破損または未保存です。") from exc
        with tarfile.open(archive, "r") as pack:
            members = pack.getmembers()
            if (len(members) != len(records) or
                {item.name for item in members} != {row["path"] for row in records} or
                any(not item.isfile() or not IMAGE_NAME.fullmatch(item.name) for item in members)):
                raise RuntimeError("保存データのアーカイブ内容がmanifestと一致しません。")
            for member in members:
                destination = local_dir / member.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                with pack.extractfile(member) as incoming, destination.open("wb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
        archive.unlink()
        atomic_json(ready_path, {"fingerprint": expected})
    return PreparedDataset(local_dir, records, expected, drive_dir)


def prepare_dataset(drive_root, scratch_root, settings, source_factory=None, exclusion=None):
    recipe = data_recipe(settings)
    excluded_ids, excluded_hashes = set(), set()
    if "exclusion" in recipe:
        if exclusion is None or stable_hash(exclusion) != recipe["exclusion"]["sha256"]:
            raise ValueError("The exact prior-image exclusion ledger is required.")
        excluded_ids = set(exclusion["image_ids"])
        excluded_hashes = set(exclusion["image_sha256"])
    elif exclusion is not None:
        raise ValueError("Image exclusions must be included in the dataset recipe.")
    if recipe["sample_count"] <= 0:
        raise ValueError("sample_countは1以上にしてください。")
    flow_shift_for_resolution(recipe["resolution"])
    key = stable_hash(recipe)[:16]
    drive_dir = Path(drive_root) / "datasets" / key
    local_dir = Path(scratch_root) / "datasets" / key
    manifest_path = drive_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        result = _restore_dataset(drive_dir, local_dir, manifest, recipe)
        if any(row["image_id"] in excluded_ids or row["image_sha256"] in excluded_hashes
               for row in result.records):
            raise RuntimeError("Saved data overlaps the prior training pools.")
        print(f"保存データを復元: {len(result):,}枚", flush=True)
        return result
    from iris3b.data.buckets import SHARED_21_512, SHARED_21_1024, closest_ratio
    buckets = SHARED_21_512 if recipe["resolution"] == 512 else SHARED_21_1024
    local_dir.mkdir(parents=True, exist_ok=True)
    (local_dir / "images").mkdir(exist_ok=True)
    records, seen, seen_hashes, skipped = [], set(), set(), 0
    source = source_factory() if source_factory else dataset_rows(settings, scratch_root)
    try:
        for row in source:
            quality = int(row["aesthetic_tier"])
            prompt = str(row.get("prompt") or "").strip()
            identity = "" if row.get("image_id") is None else str(row["image_id"])
            if (quality < recipe["min_aesthetic_tier"] or not prompt or not identity or identity in seen or
                identity in excluded_ids or
                (recipe["sfw_only"] and not _is_sfw(prompt))):
                skipped += 1
                continue
            raw = row["image"]
            image_hash = hashlib.sha256(raw).hexdigest()
            if image_hash in excluded_hashes or (exclusion is not None and image_hash in seen_hashes):
                skipped += 1
                continue
            try:
                with Image.open(io.BytesIO(raw)) as original:
                    oriented = ImageOps.exif_transpose(original)
                    oriented.load()
                    width, height = oriented.size
                    if width <= 0 or height <= 0:
                        raise ValueError("empty image")
            except (OSError, ValueError, Image.DecompressionBombError):
                skipped += 1
                continue
            if recipe["aspect_buckets"]:
                _, (target_h, target_w) = closest_ratio(height, width, buckets)
            else:
                target_h = target_w = recipe["resolution"]
            image_name = f"images/{len(records):06d}.bin"
            (local_dir / image_name).write_bytes(raw)
            records.append(dict(path=image_name, prompt=prompt, image_id=identity,
                aesthetic_tier=quality, height=target_h, width=target_w,
                original_height=height, original_width=width, image_sha256=image_hash))
            seen.add(identity)
            seen_hashes.add(image_hash)
            if len(records) % 500 == 0:
                print(f"高品質画像: {len(records):,}/{recipe['sample_count']:,}", flush=True)
            if len(records) == recipe["sample_count"]:
                break
    finally:
        if hasattr(source, "close"):
            source.close()
    if len(records) != recipe["sample_count"]:
        raise RuntimeError(f"高品質画像が不足しています: {len(records)}/{recipe['sample_count']}枚。")
    fingerprint = stable_hash({"recipe": recipe, "records": records})
    archive_name = "dataset-" + uuid.uuid4().hex[:12] + ".tar"
    archive = local_dir / archive_name
    with tarfile.open(archive, "w") as pack:
        for row in records:
            pack.add(local_dir / row["path"], arcname=row["path"], recursive=False)
    checksum = verified_copy(archive, drive_dir / archive_name)
    manifest = dict(recipe=recipe, records=records, fingerprint=fingerprint,
                    archive=archive_name, archive_sha256=checksum)
    atomic_json(manifest_path, manifest)
    atomic_json(local_dir / "ready.json", {"fingerprint": fingerprint})
    archive.unlink()
    print(f"抽出完了: {len(records):,}枚、除外{skipped:,}件。データをDriveへ保存しました。", flush=True)
    return PreparedDataset(local_dir, records, fingerprint, drive_dir)


def check_a100():
    if not torch.cuda.is_available():
        raise RuntimeError("ColabのランタイムをGPU / A100に変更してください。")
    name = torch.cuda.get_device_name(0)
    memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if "A100" not in name or memory < 35 or not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"A100・35GiB以上・BF16対応が必要です。現在: {name}, {memory:.1f}GiB")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("1台のA100を持つColabランタイムを選んでください。")
    print(f"GPU: {name} / {memory:.1f}GiB / PyTorch {torch.__version__}")


def load_models(settings, cache_root, model_dir=None, text_dir=None, device="cuda"):
    shift = flow_shift_for_resolution(settings["resolution"])
    from huggingface_hub import snapshot_download
    from omegaconf import OmegaConf
    from safetensors.torch import load_file
    from iris3b.config import inference_config
    from iris3b.models import IrisDiT
    from iris3b.text.qwen3_vl import Qwen3VLTextEncoder
    seed_all(settings["seed"])
    cache_root = Path(cache_root)
    if model_dir is None:
        model_dir = snapshot_download(BASE_REPO, revision=BASE_REVISION,
            allow_patterns=["config.yaml", "model.safetensors"], local_dir=str(cache_root / "iris-base"))
    if text_dir is None:
        text_dir = snapshot_download(TEXT_REPO, revision=TEXT_REVISION,
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
    model.to(device=device, dtype=torch.bfloat16).requires_grad_(False)
    model.activation_checkpointing = "full"
    targets = inject_lora(model, settings["rank"], settings["alpha"])
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    encoder = Qwen3VLTextEncoder(cfg.text_encoder, device=device)
    # Weights/tokenizer are already loaded from the pinned local snapshot.
    # Upstream derives the null-cache filename from pretrained; use the model
    # identity rather than a filesystem path (also valid on Windows).
    cfg.text_encoder.pretrained = TEXT_REPO
    count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"LoRA対象: {len(targets)} Linear / 学習パラメータ: {count:,} / Qwenは凍結")
    return model, encoder, cfg, targets


class LoRATrainer:
    checkpoint_tensor_key = "adapter"
    gradient_clip = 1.0
    micro_batch_size = 1

    def __init__(self, model, encoder, cfg, targets, dataset, settings, run_dir, scratch, device="cuda"):
        from iris3b.flow import RectifiedFlow
        self.model, self.encoder, self.cfg = model, encoder, cfg
        self.dataset, self.settings = dataset, dict(settings)
        self.run_dir, self.scratch, self.device = Path(run_dir), Path(scratch), torch.device(device)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.scratch.mkdir(parents=True, exist_ok=True)
        if len(dataset) != settings["sample_count"] or len(dataset) == 0:
            raise ValueError("データの画像数と学習設定が一致しません。データ準備セルから再実行してください。")
        if self.settings["grad_accum"] < 1 or self.settings["learning_rate"] <= 0:
            raise ValueError("grad_accumとlearning_rateは正の値にしてください。")
        parameters = [p for p in model.parameters() if p.requires_grad]
        self._validate_trainable_model(parameters)
        self.optimizer = self._create_optimizer(parameters)
        warmup = max(1, int(settings["warmup_steps"]))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
            lambda step: min((step + 1) / warmup, 1.0))
        self.flow = RectifiedFlow(cfg.flow, tokens=(settings["resolution"] // cfg.model.patch_size)**2)
        self.recipe = self._build_recipe(targets)
        self.signature = stable_hash(self.recipe)
        self.step, self.cursor, self.last_loss = 0, 0, None
        self.last_saved_step, self.last_saved_record = None, None
        self.invalid_optimizer_state = False
        self._epoch, self._order = None, None
        # Build the fixed null once; do not keep per-image 12-layer embeddings on disk.
        self.null = self._build_null()
        self.committed_rng = rng_state()

    def _validate_trainable_model(self, parameters):
        if not parameters or any(p.requires_grad and not name.endswith((".lora_A", ".lora_B"))
                                 for name, p in self.model.named_parameters()):
            raise ValueError("学習対象はLoRAパラメータだけにしてください。")

    def _create_optimizer(self, parameters):
        return torch.optim.AdamW(parameters, lr=self.settings["learning_rate"],
            betas=(0.9, 0.95), weight_decay=0, fused=self.device.type == "cuda")

    def _build_recipe(self, targets):
        keys = ("seed", "learning_rate", "warmup_steps", "grad_accum", "text_dropout", "rank", "alpha")
        return dict(schema=SCHEMA_VERSION, source_revision=SOURCE_REVISION,
            base_repo=BASE_REPO, base_revision=BASE_REVISION, text_repo=TEXT_REPO,
            text_revision=TEXT_REVISION, dataset_fingerprint=self.dataset.fingerprint,
            data=data_recipe(self.settings), training={key: self.settings[key] for key in keys}, targets=list(targets),
            model=asdict(self.cfg.model), flow=asdict(self.cfg.flow), base_dtype=str(next(self.model.parameters()).dtype))

    def _build_null(self):
        return self.encoder.null()

    def _checkpoint_weights(self):
        return adapter_state(self.model)

    def _restore_weights(self, state):
        load_adapter_state(self.model, state)

    def _prepare_step(self):
        pass

    def _clear_step(self):
        pass

    def _autocast(self):
        return torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda")

    def index_at(self, cursor):
        epoch, offset = divmod(cursor, len(self.dataset))
        if epoch != self._epoch:
            self._order = list(range(len(self.dataset)))
            random.Random(self.settings["seed"] + epoch).shuffle(self._order)
            self._epoch = epoch
        return self._order[offset]

    def micro_loss(self, cursor):
        image, caption = self.dataset[self.index_at(cursor)]
        image = image.unsqueeze(0).to(self.device, dtype=torch.float32)
        # The encoder's hidden states must be ordinary detached tensors: an
        # inference-mode tensor cannot be saved for the trainable text adapter.
        with torch.no_grad():
            drop = torch.rand((), device=self.device).item() < self.settings["text_dropout"]
            encoding = self.null if drop else self.encoder.encode([caption])
            embedding = encoding.embeddings.detach().to(self.device)
            mask = encoding.mask.detach().to(self.device)
        with self._autocast():
            return self.flow.training_loss(self.model, image, embedding,
                                            model_kwargs={"y_mask": mask}).loss

    def sanity_check(self):
        state = rng_state()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        try:
            loss = self.micro_loss(self.cursor)
            if not torch.isfinite(loss):
                raise RuntimeError("最小確認のlossが非有限です。")
            loss.backward()
            gradients = [p.grad for p in self.model.parameters() if p.requires_grad and p.grad is not None]
            if not gradients or not all(torch.isfinite(grad).all() for grad in gradients):
                raise RuntimeError("学習パラメータの勾配が無効です。")
            if not any(torch.count_nonzero(grad) for grad in gradients):
                raise RuntimeError("学習パラメータの勾配が全て0です。")
            if any(p.grad is not None for p in self.model.parameters() if not p.requires_grad):
                raise RuntimeError("凍結ベースに勾配が発生しています。")
            memory = torch.cuda.max_memory_allocated(self.device) / 1024**3 if self.device.type == "cuda" else 0
            print(f"forward/backward OK / loss {loss.item():.5f} / peak allocated {memory:.2f}GiB")
        finally:
            self.optimizer.zero_grad(set_to_none=True)
            restore_rng(state)

    def save(self):
        if self.invalid_optimizer_state:
            raise RuntimeError("optimizer更新が中断されたモデルは保存できません。読み込みセルから再実行してください。")
        if self.last_saved_step == self.step:
            return self.last_saved_record
        directory = self.run_dir / "checkpoints"
        directory.mkdir(parents=True, exist_ok=True)
        filename = f"step-{self.step:08d}-{uuid.uuid4().hex[:8]}.pt"
        local_file = self.scratch / filename
        payload = dict(schema=SCHEMA_VERSION, signature=self.signature, recipe=self.recipe,
            optimizer=cpu_tree(self.optimizer.state_dict()),
            scheduler=self.scheduler.state_dict(), step=self.step, cursor=self.cursor,
            rng=self.committed_rng, last_loss=self.last_loss,
            runtime=dict(torch=str(torch.__version__), device_type=self.device.type))
        payload[self.checkpoint_tensor_key] = self._checkpoint_weights()
        torch.save(payload, local_file)
        del payload
        try:
            checksum = verified_copy(local_file, directory / filename)
        finally:
            local_file.unlink(missing_ok=True)
        index_path = directory / "index.json"
        index = json.loads(index_path.read_text()) if index_path.is_file() else {"schema": SCHEMA_VERSION, "checkpoints": []}
        item = dict(file=filename, sha256=checksum, step=self.step, cursor=self.cursor)
        existing = index["checkpoints"]
        existing.append(item)
        keep = max(2, int(self.settings["keep_checkpoints"]))
        retained, removed = existing[-keep:], existing[:-keep]
        atomic_json(index_path, {"schema": SCHEMA_VERSION, "checkpoints": retained})
        self.last_saved_step, self.last_saved_record = self.step, item
        for old in removed:
            if CHECKPOINT_NAME.fullmatch(old["file"]):
                (directory / old["file"]).unlink(missing_ok=True)
        print(f"Driveへ保存: step {self.step:,} / 使用画像 {self.cursor:,}", flush=True)
        return item

    def resume(self):
        directory = self.run_dir / "checkpoints"
        index_path = directory / "index.json"
        if not index_path.is_file():
            # An uncommitted copy is not a successfully saved checkpoint.
            if any(directory.glob("step-*.pt")):
                raise RuntimeError("チェックポイントのindexがありません。保存結果を確認してください。")
            print("新規学習を開始します。")
            return 0
        try:
            index = json.loads(index_path.read_text())
            entries = index["checkpoints"]
            if index.get("schema") != SCHEMA_VERSION or not entries:
                raise ValueError("invalid index")
        except (ValueError, KeyError, TypeError) as exc:
            raise RuntimeError("チェックポイントのindexが破損しています。") from exc
        for item in reversed(entries):
            if not CHECKPOINT_NAME.fullmatch(item["file"]):
                continue
            source = directory / item["file"]
            local = self.scratch / ("resume-" + item["file"])
            try:
                verified_copy(source, local, item["sha256"])
                payload = torch.load(local, map_location="cpu", weights_only=True, mmap=os.name != "nt")
            except (OSError, RuntimeError, EOFError, ValueError, pickle.UnpicklingError) as exc:
                print(f"破損した保存を除外: step {item['step']} ({type(exc).__name__})")
                continue
            finally:
                local.unlink(missing_ok=True)
            if payload.get("signature") != self.signature or payload.get("recipe") != self.recipe:
                raise ValueError("再開設定またはデータが保存時と違います。設定を戻すかRUN_NAMEを変更してください。")
            if (payload.get("schema") != SCHEMA_VERSION or payload["step"] != item["step"] or
                payload["cursor"] != item["cursor"] or
                payload["cursor"] != payload["step"] * self.settings["grad_accum"] * self.micro_batch_size):
                raise RuntimeError("チェックポイントのstep・データ位置が一致しません。")
            self._restore_weights(payload[self.checkpoint_tensor_key])
            self.optimizer.load_state_dict(payload["optimizer"])
            self.scheduler.load_state_dict(payload["scheduler"])
            self.step, self.cursor = payload["step"], payload["cursor"]
            self.last_loss, self.committed_rng = payload["last_loss"], payload["rng"]
            self.last_saved_step, self.last_saved_record = self.step, item
            self.invalid_optimizer_state = False
            restore_rng(self.committed_rng)
            self._epoch = None
            print(f"再開: step {self.step:,} / 使用画像 {self.cursor:,}", flush=True)
            return self.step
        raise RuntimeError("正常なチェックポイントが見つかりません。保存ファイルを確認してください。")

    def update_runtime_settings(self, settings):
        frozen_keys = self._immutable_setting_keys()
        if (any(self.settings[key] != settings[key] for key in frozen_keys)
                or data_recipe(self.settings) != data_recipe(settings)):
            raise ValueError("学習設定が読み込み時から変わっています。RUN_NAMEを変更し、データとモデルを読み直してください。")
        for key in ("max_steps", "session_minutes", "save_every", "save_minutes"):
            if settings[key] <= 0:
                raise ValueError("ステップ数・時間・保存間隔は正の値にしてください。")
            self.settings[key] = settings[key]

    def _immutable_setting_keys(self):
        return ("seed", "learning_rate", "warmup_steps", "grad_accum", "text_dropout", "rank", "alpha",
                "sample_count", "resolution", "aspect_buckets", "min_aesthetic_tier", "sfw_only")

    def train(self, max_steps=None):
        if self.invalid_optimizer_state:
            raise RuntimeError("optimizer更新が中断されています。読み込みセルから再実行してください。")
        max_steps = int(max_steps if max_steps is not None else self.settings["max_steps"])
        if self.step >= max_steps:
            print(f"目標の{max_steps:,}ステップまで完了しています。")
            return
        started = last_save = time.monotonic()
        accumulation = self.settings["grad_accum"]
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        phase = "accumulation"
        try:
            while self.step < max_steps:
                loss_sum = 0.0
                phase = "accumulation"
                self._prepare_step()
                for micro in range(accumulation):
                    loss = self.micro_loss(self.cursor + micro * self.micro_batch_size)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"step {self.step}のlossが非有限です。前回保存から再開してください。")
                    (loss / accumulation).backward()
                    loss_sum += float(loss.detach())
                    del loss
                trainable = [p for p in self.model.parameters() if p.requires_grad]
                norm = torch.nn.utils.clip_grad_norm_(trainable, self.gradient_clip, error_if_nonfinite=True)
                phase = "optimizer"
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self._clear_step()
                self.step += 1
                self.cursor += accumulation * self.micro_batch_size
                self.last_loss = loss_sum / accumulation
                self.committed_rng = rng_state()
                phase = "committed"
                now = time.monotonic()
                if self.step == 1 or self.step % 10 == 0:
                    minutes = (now - started) / 60
                    print(f"step {self.step:,}/{max_steps:,} / loss {self.last_loss:.5f} / "
                          f"grad {float(norm):.3f} / {minutes:.1f}分", flush=True)
                if (self.step % self.settings["save_every"] == 0 or
                    now - last_save >= 60 * self.settings["save_minutes"]):
                    self.save()
                    last_save = time.monotonic()
                if now - started >= 60 * self.settings["session_minutes"]:
                    print("この実行の時間上限に達しました。再実行で続けられます。")
                    break
            self.save()
        except KeyboardInterrupt:
            self.optimizer.zero_grad(set_to_none=True)
            if phase != "optimizer":
                restore_rng(self.committed_rng)
                self.save()
                print("停止しました。完了したoptimizerステップまで保存済みです。")
            else:
                self.invalid_optimizer_state = True
                print("optimizer更新中に停止しました。前回の正常な保存を保持します。読み込みセルから再実行してください。")
        except Exception:
            self.optimizer.zero_grad(set_to_none=True)
            if phase != "optimizer":
                restore_rng(self.committed_rng)
                try:
                    self.save()
                except Exception as save_error:
                    print(f"今回の緊急保存に失敗しました: {type(save_error).__name__}。前回の保存から再開してください。")
            else:
                self.invalid_optimizer_state = True
            raise
        finally:
            self._clear_step()

    @torch.no_grad()
    def sample(self, prompts, steps=50, seed=42, compare_base=True):
        if self.invalid_optimizer_state:
            raise RuntimeError("optimizer更新が中断されています。読み込みセルから再実行してください。")
        from iris3b.sampling import generate
        state = rng_state()
        was_training = self.model.training
        modules = lora_modules(self.model)
        self.model.eval()
        directory = self.run_dir / "samples"
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        try:
            modes = [False, True] if compare_base else [True]
            for enabled in modes:
                for module in modules.values():
                    module.enabled = enabled
                label = f"step-{self.step:08d}" if enabled else "base"
                for index, prompt in enumerate(prompts):
                    gen = torch.Generator(device=self.device).manual_seed(seed + index)
                    with self._autocast():
                        output = generate(self.model, self.encoder, [prompt],
                            height=self.settings["resolution"], width=self.settings["resolution"],
                            steps=steps, cfg_scale=3.0, shift=self.flow.shift, generator=gen,
                            device=self.device, num_train_timesteps=self.cfg.flow.num_train_timesteps,
                            prediction=self.cfg.flow.prediction)[0]
                    pixels = output.detach().float().cpu().add(1).mul(127.5).clamp(0, 255)
                    image = Image.fromarray(pixels.permute(1, 2, 0).round().to(torch.uint8).numpy())
                    local = self.scratch / f"{label}-{index:02d}.png"
                    image.save(local)
                    target = directory / local.name
                    verified_copy(local, target)
                    paths.append(target)
            atomic_json(directory / f"step-{self.step:08d}-prompts.json",
                        dict(prompts=prompts, seed=seed, steps=steps, cfg_scale=3.0, shift=self.flow.shift))
        finally:
            for module in modules.values():
                module.enabled = True
            self.model.train(was_training)
            restore_rng(state)
        return paths


def export_adapter(model, output_dir, recipe, step):
    from safetensors.torch import save_file
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    temp = output_dir / ("adapter-" + uuid.uuid4().hex + ".partial")
    try:
        save_file(adapter_state(model), str(temp), metadata={"format": "iris3b-linear-lora-v1"})
        os.replace(temp, output_dir / "adapter.safetensors")
    finally:
        temp.unlink(missing_ok=True)
    config = copy.deepcopy(recipe)
    config.update(format="iris3b-linear-lora-v1", optimizer_step=step)
    atomic_json(output_dir / "adapter_config.json", config)
    print(f"LoRAを書き出しました: {output_dir}")
    return output_dir


def export_merged(model, cfg, output_dir):
    from omegaconf import OmegaConf
    from safetensors.torch import save_file
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights = merged_state_dict(model)
    temp = output_dir / ("model-" + uuid.uuid4().hex + ".partial")
    try:
        save_file(weights, str(temp))
        os.replace(temp, output_dir / "model.safetensors")
    finally:
        temp.unlink(missing_ok=True)
    text = asdict(cfg.text_encoder)
    text["pretrained"], text["null_embed_dir"] = TEXT_REPO, "output/pretrained_models"
    OmegaConf.save(OmegaConf.create(dict(model=asdict(cfg.model), text_encoder=text,
                                        flow=asdict(cfg.flow))), output_dir / "config.yaml")
    print(f"公式Iris形式へマージしました: {output_dir}")
    return output_dir
