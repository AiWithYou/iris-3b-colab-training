"""CPU behavior checks for full FT. These do not validate A100 CUDA execution."""
import copy
import json
from pathlib import Path

import pytest
import torch

import iris_lora_colab as core
import iris_full_ft_colab as fullft
from iris3b.config import Config
from test_iris_lora import FakeEncoder, create_dataset, make_model


class MovingEncoder(FakeEncoder):
    def __init__(self, model):
        self.model = model
        self.device = torch.device("cpu")
        self.events = []
        self.decoder = torch.nn.Linear(32, 32).requires_grad_(False)

    def to(self, device):
        self.device = torch.device(device)
        self.events.append(("move", str(self.device)))
        self.decoder.to(self.device)
        return self

    def encode(self, prompts):
        assert all(p.grad is None for p in self.model.parameters()), "Encoded text with live DiT gradients"
        self.events.append(("encode", len(prompts)))
        return super().encode(prompts)


def make_full_trainer(root, seed=5, lr=1e-3, micro_batch=1, mixed_buckets=False):
    core.seed_all(seed)
    model = make_model()
    cfg = Config()
    cfg.model = model.cfg
    cfg.flow.shift = 4.0
    settings = dict(seed=seed, learning_rate=lr, warmup_steps=2, grad_accum=3, micro_batch_size=micro_batch,
                    text_dropout=0.4, resolution=16, aspect_buckets=False,
                    sample_count=7, min_aesthetic_tier=2, sfw_only=False,
                    max_steps=6, save_every=1, save_minutes=60, session_minutes=60,
                    keep_checkpoints=2)
    encoder = MovingEncoder(model)
    dataset = create_dataset(root / "data")
    if mixed_buckets:
        for i, record in enumerate(dataset.records):
            record["width"] = 12 if i % 2 else 16
        dataset.fingerprint = "fixture-mixed-buckets-v1"
    trainer = fullft.FullFTTrainer(model, encoder, cfg, [n for n, _ in model.named_parameters()],
        dataset, settings, root / "drive", root / "scratch", device="cpu")
    return trainer


def weights(trainer):
    return {name: tensor.detach().clone() for name, tensor in trainer.model.state_dict().items()}


def test_full_ft_updates_parameters_outside_lora_and_freezes_encoder(tmp_path):
    trainer = make_full_trainer(tmp_path)
    before = weights(trainer)
    trainer.train(max_steps=2)
    assert all(p.requires_grad and p.dtype == torch.float32 for p in trainer.model.parameters())
    changed = [name for name, tensor in trainer.model.state_dict().items() if not torch.equal(before[name], tensor)]
    assert any(name.startswith("t_embedder.") for name in changed)
    assert any("modulation" in name for name in changed)
    assert not core.lora_modules(trainer.model)
    assert all(p.grad is None and not p.requires_grad for p in trainer.encoder.decoder.parameters())
    assert trainer.encoder.device.type == "cpu" and not trainer._conditionings
    assert [event for event in trainer.encoder.events if event == ("encode", 3)] == [("encode", 3)] * 2
    index = json.loads((trainer.run_dir / "checkpoints/index.json").read_text())
    saved = torch.load(trainer.run_dir / "checkpoints" / index["checkpoints"][-1]["file"], weights_only=True)
    assert "model" in saved and "adapter" not in saved
    assert saved["recipe"]["training_mode"] == "full-iris-fp32-v1"
    assert set(saved["model"]) == set(before)


def test_full_resume_matches_uninterrupted_including_all_model_weights(tmp_path):
    uninterrupted = make_full_trainer(tmp_path / "full")
    uninterrupted.train(max_steps=6)
    expected = weights(uninterrupted)
    split = make_full_trainer(tmp_path / "split")
    split.train(max_steps=3)
    resumed = make_full_trainer(tmp_path / "split")
    assert resumed.resume() == 3
    resumed.train(max_steps=6)
    assert resumed.cursor == uninterrupted.cursor == 18
    assert all(torch.equal(expected[name], tensor) for name, tensor in resumed.model.state_dict().items())
    assert resumed.scheduler.state_dict() == uninterrupted.scheduler.state_dict()


def test_bucket_batches_cover_every_image_and_keep_shapes_across_epochs():
    records = [dict(height=16, width=12 if i % 3 else 16) for i in range(11)]
    order = fullft.BucketBatchOrder(records, batch_size=4, seed=42)
    assert order.epoch_size == 12 and order.padding == 1
    assert order.order(0) == fullft.BucketBatchOrder(records, 4, 42).order(0)
    assert order.order(0) != order.order(1)
    for epoch in range(4):
        indices = order.order(epoch)
        assert set(indices) == set(range(len(records)))
        for offset in range(0, len(indices), 4):
            assert len({(records[i]["height"], records[i]["width"])
                        for i in indices[offset:offset + 4]}) == 1


def test_real_micro_batches_resume_exactly_across_bucket_epoch_boundaries(tmp_path):
    full = make_full_trainer(tmp_path / "full-batch", micro_batch=2, mixed_buckets=True)
    observed_shapes = []
    original = full.flow.training_loss

    def observe(model, images, text, **kwargs):
        observed_shapes.append((tuple(images.shape), text.shape[0], kwargs["model_kwargs"]["y_mask"].shape[0]))
        return original(model, images, text, **kwargs)

    full.flow.training_loss = observe
    full.train(max_steps=6)
    assert full.cursor == 36 and full.bucket_order.epoch_size == 8
    assert len(observed_shapes) == 18
    assert all(shape[0] == text_batch == mask_batch == 2 for shape, text_batch, mask_batch in observed_shapes)
    assert {shape[-1] for shape, _, _ in observed_shapes} == {12, 16}
    split = make_full_trainer(tmp_path / "split-batch", micro_batch=2, mixed_buckets=True)
    split.train(max_steps=3)
    resumed = make_full_trainer(tmp_path / "split-batch", micro_batch=2, mixed_buckets=True)
    assert resumed.resume() == 3 and resumed.cursor == 18
    resumed.train(max_steps=6)
    assert all(torch.equal(tensor, resumed.model.state_dict()[name]) for name, tensor in weights(full).items())
    assert full.scheduler.state_dict() == resumed.scheduler.state_dict()
    assert [event for event in full.encoder.events if event == ("encode", 6)] == [("encode", 6)] * 6


def test_micro_batch_size_is_immutable_and_checkpoint_mismatch_is_rejected(tmp_path):
    trainer = make_full_trainer(tmp_path, micro_batch=2)
    trainer.train(max_steps=1)
    changed = make_full_trainer(tmp_path, micro_batch=3)
    with pytest.raises(ValueError, match="設定"):
        changed.resume()
    settings = dict(trainer.settings, micro_batch_size=3)
    with pytest.raises(ValueError, match="設定"):
        trainer.update_runtime_settings(settings)


def test_interrupted_real_batch_discards_partial_accumulation_and_replays_exactly(tmp_path):
    reference = make_full_trainer(tmp_path / "batch-reference", micro_batch=2, mixed_buckets=True)
    reference.train(max_steps=2)
    stopped = make_full_trainer(tmp_path / "batch-split", micro_batch=2, mixed_buckets=True)
    original = stopped.micro_loss
    calls = 0

    def interrupt(cursor):
        nonlocal calls
        calls += 1
        if calls == 5:
            raise KeyboardInterrupt
        return original(cursor)

    stopped.micro_loss = interrupt
    stopped.train(max_steps=2)
    assert stopped.step == 1 and stopped.cursor == 6
    resumed = make_full_trainer(tmp_path / "batch-split", micro_batch=2, mixed_buckets=True)
    assert resumed.resume() == 1
    resumed.train(max_steps=2)
    assert all(torch.equal(tensor, resumed.model.state_dict()[name]) for name, tensor in weights(reference).items())


def test_full_interrupted_accumulation_rolls_back_conditioning_and_rng(tmp_path):
    reference = make_full_trainer(tmp_path / "reference")
    reference.train(max_steps=2)
    stopped = make_full_trainer(tmp_path / "split")
    original = stopped.micro_loss
    calls = 0

    def interrupt(cursor):
        nonlocal calls
        calls += 1
        if calls == 5:
            raise KeyboardInterrupt
        return original(cursor)

    stopped.micro_loss = interrupt
    stopped.train(max_steps=2)
    assert stopped.step == 1 and stopped.cursor == 3
    assert not stopped._conditionings
    assert all(p.grad is None for p in stopped.model.parameters())
    resumed = make_full_trainer(tmp_path / "split")
    assert resumed.resume() == 1
    resumed.train(max_steps=2)
    assert all(torch.equal(tensor, resumed.model.state_dict()[name]) for name, tensor in weights(reference).items())


def test_full_corrupt_checkpoint_fallback_and_configuration_guard(tmp_path):
    trained = make_full_trainer(tmp_path)
    trained.train(max_steps=2)
    index = json.loads((trained.run_dir / "checkpoints/index.json").read_text())
    (trained.run_dir / "checkpoints" / index["checkpoints"][-1]["file"]).write_bytes(b"truncated")
    restored = make_full_trainer(tmp_path)
    assert restored.resume() == 1
    changed = make_full_trainer(tmp_path, lr=0.002)
    with pytest.raises(ValueError, match="設定"):
        changed.resume()


def test_full_sanity_restores_rng_without_training_or_resident_conditions(tmp_path):
    trainer = make_full_trainer(tmp_path)
    state, before = core.rng_state(), weights(trainer)
    trainer.sanity_check()
    assert trainer.step == 0 and trainer.cursor == 0
    assert torch.equal(state["torch"], core.rng_state()["torch"])
    assert all(torch.equal(tensor, trainer.model.state_dict()[name]) for name, tensor in before.items())
    assert not trainer._conditionings and trainer.encoder.device.type == "cpu"


def test_full_export_has_official_keys_bf16_and_is_inference_loadable(tmp_path):
    from safetensors.torch import load_file
    from omegaconf import OmegaConf
    trainer = make_full_trainer(tmp_path)
    trainer.train(max_steps=1)
    output = fullft.export_full_model(trainer, tmp_path / "export")
    exported = load_file(str(output / "model.safetensors"))
    assert exported.keys() == trainer.model.state_dict().keys()
    assert all(t.dtype == torch.bfloat16 for t in exported.values() if t.is_floating_point())
    plain = make_model().to(torch.bfloat16).eval()
    plain.load_state_dict(exported, strict=True)
    control = copy.deepcopy(trainer.model).to(torch.bfloat16).eval()
    x = torch.randn(1, 3, 16, 16)
    y = trainer.encoder.encode(["illustration"]).embeddings
    mask = torch.ones(1, 8, dtype=torch.long)
    t = torch.tensor([400.0])
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        assert torch.allclose(plain(x, t, y, y_mask=mask).x, control(x, t, y, y_mask=mask).x, atol=1e-5)
    cfg = OmegaConf.load(output / "config.yaml")
    assert cfg.text_encoder.pretrained == core.TEXT_REPO
    manifest = json.loads((output / "export_manifest.json").read_text())
    assert manifest["files"]["model.safetensors"] == core.file_hash(output / "model.safetensors")
    assert manifest["optimizer_step"] == 1
    trainer.invalid_optimizer_state = True
    with pytest.raises(RuntimeError, match="正常"):
        fullft.export_full_model(trainer, tmp_path / "invalid-export")


def test_full_rejects_partial_freezing_and_any_lora(tmp_path):
    trainer = make_full_trainer(tmp_path)
    next(trainer.model.parameters()).requires_grad_(False)
    with pytest.raises(ValueError, match="全パラメータ"):
        trainer._validate_trainable_model(list(trainer.model.parameters()))
    trainer.model.requires_grad_(True)
    core.inject_lora(trainer.model, rank=2, alpha=2)
    trainer.model.requires_grad_(True)
    with pytest.raises(ValueError, match="LoRAなし"):
        trainer._validate_trainable_model(list(trainer.model.parameters()))


def test_full_sampling_preserves_rng_and_returns_one_image_per_prompt(tmp_path):
    trainer = make_full_trainer(tmp_path)
    trainer.train(max_steps=1)
    state, before = core.rng_state(), weights(trainer)
    paths = trainer.sample(["anime illustration"], steps=2, seed=17)
    assert len(paths) == 1 and paths[0].is_file()
    assert torch.equal(state["torch"], core.rng_state()["torch"])
    assert trainer.step == 1 and trainer.model.training and trainer.encoder.device.type == "cpu"
    assert all(torch.equal(tensor, trainer.model.state_dict()[name]) for name, tensor in before.items())


def test_real_bnb_quantized_state_survives_safe_cpu_serialization(tmp_path):
    """Exercise real bnb save/load on CPU; no CUDA optimizer kernel is run."""
    from bitsandbytes.optim import AdamW8bit
    p = torch.nn.Parameter(torch.zeros(8192))
    optimizer = AdamW8bit([p], lr=1e-5, betas=(0.9, 0.95), weight_decay=0)
    state = dict(step=3, state1=torch.arange(8192).to(torch.uint8),
                 state2=torch.arange(8192).flip(0).to(torch.uint8),
                 qmap1=torch.linspace(-1, 1, 256), qmap2=torch.linspace(0, 1, 256),
                 absmax1=torch.ones(32), absmax2=torch.full((32,), 2.0))
    optimizer.state[p] = state
    checkpoint = tmp_path / "8bit-state.pt"
    torch.save(core.cpu_tree(optimizer.state_dict()), checkpoint)
    q = torch.nn.Parameter(p.detach().clone())
    restored = AdamW8bit([q], lr=1e-5, betas=(0.9, 0.95), weight_decay=0)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    for name, tensor in state.items():
        actual = restored.state[q][name]
        if isinstance(tensor, torch.Tensor):
            assert torch.equal(tensor, actual) and tensor.dtype == actual.dtype
        else:
            assert tensor == actual


def test_full_notebook_is_standalone_and_has_no_saved_outputs():
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "Iris3B_FullFT_A100.ipynb").read_text(encoding="utf-8"))
    embedded = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        assert cell["outputs"] == [] and cell["execution_count"] is None
        source = "".join(cell["source"])
        if source.startswith("%%writefile"):
            filename = source.splitlines()[0].rsplit("/", 1)[1]
            embedded[filename] = source.split("\n", 1)[1]
            compile(embedded[filename], filename, "exec")
        else:
            compile(source, "fullft_notebook_cell", "exec")
    assert set(embedded) == {"iris_lora_colab.py", "iris_full_ft_colab.py"}
    assert all(source == (root / name).read_text(encoding="utf-8") for name, source in embedded.items())
