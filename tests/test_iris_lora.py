"""Behavioral checks against the pinned, actual Iris architecture."""
import copy
import ast
import errno
import io
import json
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import iris_lora_colab as app
from iris3b.config import Config, ModelConfig, PixelStageConfig
from iris3b.models import IrisDiT
from iris3b.text.base import TextEncoding

torch.set_num_threads(1)


class FakeEncoder:
    def encode(self, prompts):
        rows = []
        for prompt in prompts:
            gen = torch.Generator().manual_seed(sum(prompt.encode()) + 17)
            rows.append(torch.randn(8, 2, 32, generator=gen))
        return TextEncoding(torch.stack(rows), torch.ones(len(prompts), 8, dtype=torch.long))

    def null(self, negative_prompt=""):
        return self.encode([negative_prompt])


def make_model(device="cpu"):
    cfg = ModelConfig(hidden_size=32, depth=2, dual_depth=1, num_heads=4,
        num_kv_heads=2, patch_size=4, text_dim=32, text_len=8,
        text_lap_num_layers=2, text_lap_num_heads=4, adaln_zero_init=False,
        pixel=PixelStageConfig(depth=1, hidden_size=4, attn_hidden_size=32, num_heads=4))
    model = IrisDiT(cfg)
    torch.nn.init.normal_(model.final_layer.linear.weight, std=0.1)
    model.activation_checkpointing = "full"
    return model.to(device)


def create_dataset(path, size=7):
    path.mkdir(parents=True, exist_ok=True)
    records = []
    for index in range(size):
        name = f"images/{index:06d}.avif"
        (path / "images").mkdir(exist_ok=True)
        Image.new("RGB", (20 + index, 24), (30 * index, 20, 80)).save(path / name, format="PNG")
        records.append(dict(path=name, prompt=f"illustration {index}", image_id=str(index),
                            aesthetic_tier=3, height=16, width=16))
    return app.PreparedDataset(path, records, "fixture-dataset-v1")


def make_trainer(root, seed=5, device="cpu", lr=1e-3):
    app.seed_all(seed)
    model = make_model(device)
    if device == "cuda":
        model.to(dtype=torch.bfloat16)
    targets = app.inject_lora(model, rank=2, alpha=2)
    cfg = Config()
    cfg.model = model.cfg
    cfg.flow.shift = 3.0
    settings = dict(seed=seed, learning_rate=lr, warmup_steps=2, grad_accum=3,
                    text_dropout=0.4, rank=2, alpha=2, resolution=16,
                    aspect_buckets=False, sample_count=7, min_aesthetic_tier=2,
                    sfw_only=False, max_steps=6, save_every=1, save_minutes=60,
                    session_minutes=60, keep_checkpoints=3)
    dataset = create_dataset(root / "data")
    trainer = app.LoRATrainer(model, FakeEncoder(), cfg, targets, dataset, settings,
                             root / "drive", root / "scratch", device=device)
    return trainer


def test_lora_is_identity_until_trained_and_merge_matches():
    app.seed_all(8)
    model = make_model().eval()
    plain = copy.deepcopy(model)
    targets = app.inject_lora(model, rank=2, alpha=2)
    x = torch.randn(1, 3, 16, 16)
    y = torch.randn(1, 8, 2, 32)
    mask = torch.ones(1, 8, dtype=torch.long)
    t = torch.tensor([500.0])
    with torch.no_grad():
        initial = model(x, t, y, y_mask=mask).x
        for module in app.lora_modules(model).values():
            module.enabled = False
        assert torch.equal(initial, model(x, t, y, y_mask=mask).x)
        # Separate weight allocations may pick different CPU GEMM kernels.
        assert torch.allclose(initial, plain(x, t, y, y_mask=mask).x, atol=1e-6)
        for module in app.lora_modules(model).values():
            module.enabled = True
    for module in app.lora_modules(model).values():
        assert not module.base.weight.requires_grad
        torch.nn.init.normal_(module.lora_B, std=0.02)
    plain.load_state_dict(app.merged_state_dict(model), strict=True)
    assert torch.allclose(model(x, t, y, y_mask=mask).x, plain(x, t, y, y_mask=mask).x, atol=1e-5)
    assert targets and all("adaln" not in target for target in targets)


def test_resume_matches_uninterrupted_across_dataset_epochs(tmp_path):
    full = make_trainer(tmp_path / "full")
    full.train(max_steps=6)
    expected = app.adapter_state(full.model)
    split = make_trainer(tmp_path / "split")
    split.train(max_steps=3)
    resumed = make_trainer(tmp_path / "split")
    assert resumed.resume() == 3
    resumed.train(max_steps=6)
    assert resumed.cursor == full.cursor == 18
    assert resumed.scheduler.state_dict() == full.scheduler.state_dict()
    actual = app.adapter_state(resumed.model)
    assert all(torch.equal(expected[k], actual[k]) for k in expected)
    assert all(p.grad is None for n, p in resumed.model.named_parameters() if ".lora_" not in n)
    assert any(torch.count_nonzero(v) for k, v in expected.items() if k.endswith("lora_B"))


def test_corrupt_newest_checkpoint_falls_back(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.train(max_steps=2)
    index = json.loads((tmp_path / "drive/checkpoints/index.json").read_text())
    newest = index["checkpoints"][-1]
    (tmp_path / "drive/checkpoints" / newest["file"]).write_bytes(b"truncated")
    resumed = make_trainer(tmp_path)
    assert resumed.resume() == 1
    assert resumed.cursor == 3


def test_all_corrupt_checkpoints_do_not_silently_restart(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.train(max_steps=1)
    index = json.loads((tmp_path / "drive/checkpoints/index.json").read_text())
    for item in index["checkpoints"]:
        (tmp_path / "drive/checkpoints" / item["file"]).write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="正常なチェックポイント"):
        make_trainer(tmp_path).resume()


def test_changed_learning_rate_is_rejected_before_loading(tmp_path):
    make_trainer(tmp_path).train(max_steps=1)
    with pytest.raises(ValueError, match="設定"):
        make_trainer(tmp_path, lr=2e-3).resume()


def test_stop_inside_accumulation_restores_committed_rng_and_cursor(tmp_path):
    full = make_trainer(tmp_path / "full")
    full.train(max_steps=3)
    split = make_trainer(tmp_path / "split")
    original = split.micro_loss
    calls = 0
    def interrupt(cursor):
        nonlocal calls
        calls += 1
        if calls == 5:
            raise KeyboardInterrupt()
        return original(cursor)
    split.micro_loss = interrupt
    split.train(max_steps=3)
    assert split.step == 1 and split.cursor == 3
    resumed = make_trainer(tmp_path / "split")
    assert resumed.resume() == 1
    resumed.train(max_steps=3)
    expected, actual = app.adapter_state(full.model), app.adapter_state(resumed.model)
    assert all(torch.equal(expected[k], actual[k]) for k in expected)


def test_stop_during_optimizer_does_not_publish_partial_update(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.train(max_steps=1)
    def interrupted_step():
        with torch.no_grad():
            next(p for p in trainer.model.parameters() if p.requires_grad).add_(100)
        raise KeyboardInterrupt()
    trainer.optimizer.step = interrupted_step
    trainer.train(max_steps=2)
    with pytest.raises(RuntimeError, match="中断"):
        trainer.train(max_steps=2)
    resumed = make_trainer(tmp_path)
    assert resumed.resume() == 1


def test_changed_settings_are_not_silently_ignored(tmp_path):
    trainer = make_trainer(tmp_path)
    changed = dict(trainer.settings, learning_rate=0.5)
    with pytest.raises(ValueError, match="設定"):
        trainer.update_runtime_settings(changed)


def test_samples_preserve_training_rng_and_checkpoint_step(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.train(max_steps=1)
    state = app.rng_state()
    paths = trainer.sample(["anime village"], steps=2, compare_base=True)
    after = app.rng_state()
    assert len(paths) == 2 and all(path.is_file() for path in paths)
    assert torch.equal(state["torch"], after["torch"])
    assert state["python"] == after["python"]
    assert trainer.step == 1 and trainer.model.training


@pytest.mark.parametrize("resolution", [512, 1024])
def test_dataset_selection_archive_and_restore(tmp_path, resolution):
    def row(index, quality=3):
        buffer = io.BytesIO()
        Image.new("RGB", (40, 70), (index, 20, 40)).save(buffer, format="PNG")
        return dict(image_id=str(index), image=buffer.getvalue(), prompt="1girl, safe, blue hair",
                    aesthetic_tier=quality)
    rows = [row(1, 1), row(2), row(2), dict(row(3), image=b"bad"), row(4), row(5)]
    settings = dict(seed=5, resolution=resolution, aspect_buckets=True, sample_count=3,
                    min_aesthetic_tier=2, sfw_only=False)
    prepared = app.prepare_dataset(tmp_path / "drive", tmp_path / "local", settings,
                                   source_factory=lambda: iter(rows))
    assert len(prepared) == 3
    assert {row["image_id"] for row in prepared.records} == {"2", "4", "5"}
    image, caption = prepared[0]
    assert image.shape == (3, 672 * resolution // 512, 384 * resolution // 512)
    assert image.shape[1] % 16 == image.shape[2] % 16 == 0
    assert caption
    restored = app.prepare_dataset(tmp_path / "drive", tmp_path / "other", settings,
                                  source_factory=lambda: pytest.fail("downloaded despite saved data"))
    assert restored.fingerprint == prepared.fingerprint
    assert torch.equal(restored[0][0], image)
    metadata = json.loads((restored.drive_dir / "manifest.json").read_text())
    (restored.drive_dir / metadata["archive"]).write_bytes(b"corrupted archive")
    with pytest.raises(RuntimeError, match="データ"):
        app.prepare_dataset(tmp_path / "drive", tmp_path / "third", settings)


def test_drive_style_fsync_unsupported_keeps_checksum_verification(tmp_path, monkeypatch):
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.write_bytes(b"saved data")
    def unavailable(_):
        raise OSError(errno.ENOTSUP, "fsync unsupported")
    monkeypatch.setattr(app.os, "fsync", unavailable)
    app.verified_copy(source, destination)
    app.atomic_json(tmp_path / "index.json", {"step": 1})
    assert destination.read_bytes() == source.read_bytes()
    assert json.loads((tmp_path / "index.json").read_text()) == {"step": 1}


def test_real_io_failure_does_not_publish_copy(tmp_path, monkeypatch):
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.write_bytes(b"saved data")
    destination.write_bytes(b"previous good data")
    def disk_full(_):
        raise OSError(errno.ENOSPC, "disk full")
    monkeypatch.setattr(app.os, "fsync", disk_full)
    with pytest.raises(OSError):
        app.verified_copy(source, destination)
    assert destination.read_bytes() == b"previous good data"


def test_parquet_source_uses_binary_images_and_closes_early(tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    import huggingface_hub
    rows = [dict(image_id=str(i), image=b"raw-avif", prompt="anime illustration", aesthetic_tier=3)
            for i in range(100)]
    template = tmp_path / "template.parquet"
    downloaded = tmp_path / "downloaded.parquet"
    pq.write_table(pa.Table.from_pylist(rows), template, row_group_size=25)
    class FakeApi:
        def list_repo_files(self, *args, **kwargs):
            return ["metadata.json", "data_shard_00000.parquet"]
    def download(*args, **kwargs):
        assert kwargs["revision"] == "a" * 40
        downloaded.write_bytes(template.read_bytes())
        return str(downloaded)
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    stream = app.dataset_rows({"seed": 42, "dataset_repo": "owner/images", "dataset_revision": "a" * 40,
        "dataset_columns": {"image_id": "image_id", "aesthetic_tier": "aesthetic_tier"}}, tmp_path)
    row = next(stream)
    assert row["image"] == b"raw-avif" and row["aesthetic_tier"] == 3
    stream.close()
    assert not downloaded.exists()


def test_missing_dataset_source_fails_before_downloading(tmp_path):
    with pytest.raises(ValueError, match="DATASET_REPO"):
        next(app.dataset_rows({"seed": 42}, tmp_path))


@pytest.mark.parametrize("resolution", [512, 1024])
def test_actual_avif_image_can_be_selected_and_decoded(tmp_path, resolution):
    if not app.features.check("avif"):
        pytest.skip("AVIF decoder unavailable")
    buffer = io.BytesIO()
    Image.new("RGB", (80, 140), (80, 100, 160)).save(buffer, format="AVIF")
    settings = dict(seed=5, resolution=resolution, aspect_buckets=True, sample_count=1,
                    min_aesthetic_tier=2, sfw_only=False)
    rows = [dict(image_id="avif-fixture", image=buffer.getvalue(), prompt="anime illustration",
                 aesthetic_tier=3)]
    prepared = app.prepare_dataset(tmp_path / "drive", tmp_path / "local", settings,
                                   source_factory=lambda: iter(rows))
    image, caption = prepared[0]
    assert image.shape == (3, 672 * resolution // 512, 384 * resolution // 512)
    assert torch.isfinite(image).all() and -1 <= image.min() <= image.max() <= 1


def test_adapter_export_is_loadable_and_contains_only_lora(tmp_path):
    from safetensors.torch import load_file
    trainer = make_trainer(tmp_path)
    trainer.train(max_steps=1)
    target = app.export_adapter(trainer.model, tmp_path / "export", trainer.recipe, trainer.step)
    weights = load_file(str(target / "adapter.safetensors"))
    assert weights and all(name.endswith((".lora_A", ".lora_B")) for name in weights)
    assert json.loads((target / "adapter_config.json").read_text())["base_revision"] == app.BASE_REVISION


def test_full_3b_architecture_lora_targets_on_meta_device():
    with torch.device("meta"):
        model = IrisDiT(ModelConfig())
        targets = app.inject_lora(model, rank=16, alpha=16)
    assert len(targets) > 200
    assert any(name.startswith("pixel_blocks.") for name in targets)
    assert any(name.startswith("y_embedder.") for name in targets)
    assert all("modulation_cores" not in name and "adaln" not in name for name in targets)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) < 100_000_000


def test_local_text_snapshot_does_not_enter_null_cache_filename(tmp_path, monkeypatch):
    from iris3b.text.qwen3_vl import Qwen3VLTextEncoder
    import iris3b.text.qwen3_vl as qwen_module
    from safetensors.torch import save_file
    model = make_model()
    cfg = Config()
    cfg.model = model.cfg
    cfg.text_encoder.dim = 32
    cfg.text_encoder.hidden_layers = [1, 2]
    base_dir, text_dir = tmp_path / "base", tmp_path / "text"
    base_dir.mkdir()
    text_dir.mkdir()
    from dataclasses import asdict
    from omegaconf import OmegaConf
    OmegaConf.save(OmegaConf.create(asdict(cfg)), base_dir / "config.yaml")
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(base_dir / "model.safetensors"))
    class FakeQwen(Qwen3VLTextEncoder):
        def __init__(self, config, device):
            self.cfg = config
            self.loaded_from = config.pretrained
            self.dim, self.max_length = config.dim, config.max_length
            self.hidden_layers = tuple(config.hidden_layers)
    monkeypatch.setattr(qwen_module, "Qwen3VLTextEncoder", FakeQwen)
    _, encoder, loaded_cfg, _ = app.load_models(dict(seed=42, rank=2, alpha=2, resolution=512),
        tmp_path / "cache", model_dir=base_dir, text_dir=text_dir, device="cpu")
    assert encoder.loaded_from == str(text_dir)
    assert loaded_cfg.text_encoder.pretrained == app.TEXT_REPO
    name = encoder._null_cache_path().name
    assert ":" not in name and "\\" not in name and "Qwen" in name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_bf16_frozen_base_fp32_lora_backward_on_gpu(tmp_path):
    trainer = make_trainer(tmp_path, device="cuda")
    trainer.sanity_check()
    trainer.train(max_steps=1)
    assert trainer.step == 1
    assert all(p.dtype == torch.float32 for p in trainer.model.parameters() if p.requires_grad)
    assert all(torch.isfinite(p).all() for p in trainer.model.parameters() if p.requires_grad)
    resumed = make_trainer(tmp_path, device="cuda")
    assert resumed.resume() == 1
    resumed.train(max_steps=2)
    assert resumed.step == 2 and resumed.cursor == 6


@pytest.mark.parametrize("filename,resolution,rank", [
    ("Iris3B_LoRA_A100.ipynb", 1024, 32),
    ("Iris3B_LoRA_A100_512.ipynb", 512, 16),
])
def test_notebook_is_self_contained_and_code_cells_parse(filename, resolution, rank):
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / filename).read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    embedded = None
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        assert not cell["outputs"] and cell["execution_count"] is None
        code = "".join(cell["source"])
        if code.startswith("%%writefile"):
            embedded = code.split("\n", 1)[1]
            compile(embedded, "embedded_helper", "exec")
        else:
            compile(code, "notebook_cell", "exec")
            if code.startswith("#@title 学習設定"):
                statements = ast.parse(code).body
                values = {node.targets[0].id: ast.literal_eval(node.value)
                          for node in statements if isinstance(node, ast.Assign)
                          and isinstance(node.targets[0], ast.Name)
                          and node.targets[0].id in ("RUN_NAME", "RESOLUTION", "LORA_RANK", "LORA_ALPHA")}
                assert values == dict(RUN_NAME=f"iris3b-lora-{resolution}-r{rank}",
                                      RESOLUTION=resolution, LORA_RANK=rank, LORA_ALPHA=rank)
                settings = next(node.value for node in statements if isinstance(node, ast.Assign)
                                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "settings")
                size = next(item.value for item in settings.keywords if item.arg == "resolution")
                assert isinstance(size, ast.Name) and size.id == "RESOLUTION"
    assert embedded == (root / "iris_lora_colab.py").read_text(encoding="utf-8")


def test_unsupported_resolution_is_rejected_before_data_or_model_download(tmp_path):
    settings = dict(seed=5, resolution=768, aspect_buckets=True, sample_count=1,
                    min_aesthetic_tier=2, sfw_only=False)
    with pytest.raises(ValueError, match="512または1024"):
        app.prepare_dataset(tmp_path / "drive", tmp_path / "local", settings,
                            source_factory=lambda: pytest.fail("Downloaded unsupported resolution"))
    with pytest.raises(ValueError, match="512または1024"):
        app.load_models(settings, tmp_path / "cache")
    assert not (tmp_path / "cache").exists()
    assert app.flow_shift_for_resolution(512) == 3.0
    assert app.flow_shift_for_resolution(1024) == 4.0
