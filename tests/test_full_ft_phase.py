"""New-data continuation keeps learned state and resumes at the correct phase position."""
import copy
import json

import pytest
import torch

import iris_lora_colab as core
from iris_full_ft_phase_colab import PhaseFTTrainer, state_sample_digest
from test_full_ft import MovingEncoder, make_full_trainer, weights
from test_iris_lora import create_dataset, make_model


def parent_checkpoint(root):
    parent = make_full_trainer(root, micro_batch=2, mixed_buckets=True)
    parent.train(max_steps=3)
    index = json.loads((parent.run_dir / "checkpoints/index.json").read_text())
    return parent, index["checkpoints"][-1]


def new_phase(root, parent, record, lr=None, transition=None):
    core.seed_all(parent.settings["seed"])
    model = make_model()
    dataset = create_dataset(root / "new-data", size=11)
    for i, row in enumerate(dataset.records):
        row["width"] = 12 if i % 2 else 16
        row["prompt"] = "new highest-quality phase " + str(i)
    dataset.fingerprint = "new-tier3-dataset-v1"
    settings = dict(parent.settings, sample_count=11, min_aesthetic_tier=3, max_steps=8)
    if lr is not None:
        settings["learning_rate"] = lr
    return PhaseFTTrainer(model, MovingEncoder(model), copy.deepcopy(parent.cfg),
        [n for n, _ in model.named_parameters()], dataset, settings,
        root / "drive", root / "scratch", device="cpu", parent_record=record,
        learning_rate_transition=transition)


def test_parent_state_retained_and_new_data_starts_from_zero(tmp_path):
    parent, record = parent_checkpoint(tmp_path / "parent")
    phase = new_phase(tmp_path / "phase", parent, record)
    report = phase.start_from_parent(parent.run_dir)
    assert phase.step == 3 and phase.cursor == 18
    assert all(torch.equal(t, phase.model.state_dict()[name]) for name, t in weights(parent).items())
    assert state_sample_digest(phase.optimizer.state_dict()) == state_sample_digest(parent.optimizer.state_dict())
    assert phase.scheduler.state_dict() == parent.scheduler.state_dict()
    assert phase.index_at(phase.cursor) == phase.bucket_order.order(0)[0]
    assert report["new_dataset_cursor"] == 0 and report["rng_verified"]
    assert phase.save() is None
    assert not (phase.run_dir / "checkpoints/index.json").exists()
    assert core.file_hash(parent.run_dir / "checkpoints" / record["file"]) == record["sha256"]


def test_new_phase_resume_matches_uninterrupted_across_new_data_epoch(tmp_path):
    parent, record = parent_checkpoint(tmp_path / "parent")
    full = new_phase(tmp_path / "full", parent, record)
    full.start_from_parent(parent.run_dir)
    full.train(max_steps=8)
    split = new_phase(tmp_path / "split", parent, record)
    split.start_from_parent(parent.run_dir)
    split.train(max_steps=5)
    resumed = new_phase(tmp_path / "split", parent, record)
    assert resumed.resume() == 5
    assert resumed.cursor - resumed.phase_start_cursor == 12
    resumed.train(max_steps=8)
    assert resumed.cursor == full.cursor == 48
    assert resumed.cursor - resumed.phase_start_cursor == 30
    assert all(torch.equal(t, resumed.model.state_dict()[name]) for name, t in weights(full).items())
    assert state_sample_digest(resumed.optimizer.state_dict()) == state_sample_digest(full.optimizer.state_dict())
    assert resumed.scheduler.state_dict() == full.scheduler.state_dict()
    assert (parent.run_dir / "checkpoints" / record["file"]).is_file()


def test_parent_training_changes_rejected_before_loading(tmp_path):
    parent, record = parent_checkpoint(tmp_path / "parent")
    phase = new_phase(tmp_path / "phase", parent, record, lr=parent.settings["learning_rate"] * 2)
    initial = weights(phase)
    with pytest.raises(ValueError, match="conditions must match"):
        phase.start_from_parent(parent.run_dir)
    assert all(torch.equal(t, phase.model.state_dict()[name]) for name, t in initial.items())


def test_phase_resume_rejects_different_parent_identity(tmp_path):
    parent, record = parent_checkpoint(tmp_path / "parent")
    phase = new_phase(tmp_path / "phase", parent, record)
    phase.start_from_parent(parent.run_dir)
    phase.train(max_steps=4)
    changed = dict(record, sha256="0" * 64)
    other = new_phase(tmp_path / "phase", parent, changed)
    with pytest.raises(ValueError, match="再開設定"):
        other.resume()


def test_explicit_lr_increase_keeps_moments_and_survives_steps_and_resume(tmp_path):
    parent, record = parent_checkpoint(tmp_path / "parent")
    old_lr = parent.settings["learning_rate"]
    new_lr = old_lr * 5
    transition = {"from": old_lr, "to": new_lr}
    parent_moments = state_sample_digest(parent.optimizer.state_dict()["state"])
    original_epoch = parent.scheduler.state_dict()["last_epoch"]
    full = new_phase(tmp_path / "full-lr", parent, record, lr=new_lr, transition=transition)
    receipt = full.start_from_parent(parent.run_dir)
    assert receipt["optimizer_moments_retained"]
    assert state_sample_digest(full.optimizer.state_dict()["state"]) == parent_moments
    assert full.scheduler.state_dict()["last_epoch"] == original_epoch
    assert full.optimizer.param_groups[0]["lr"] == new_lr
    full.train(max_steps=8)
    assert full.optimizer.param_groups[0]["lr"] == new_lr
    split = new_phase(tmp_path / "split-lr", parent, record, lr=new_lr, transition=transition)
    split.start_from_parent(parent.run_dir)
    split.train(max_steps=5)
    resumed = new_phase(tmp_path / "split-lr", parent, record, lr=new_lr, transition=transition)
    assert resumed.resume() == 5
    assert resumed.optimizer.param_groups[0]["lr"] == new_lr
    resumed.train(max_steps=8)
    assert resumed.optimizer.param_groups[0]["lr"] == new_lr
    assert all(torch.equal(t, resumed.model.state_dict()[name]) for name, t in weights(full).items())
    assert state_sample_digest(resumed.optimizer.state_dict()) == state_sample_digest(full.optimizer.state_dict())
    assert resumed.scheduler.state_dict() == full.scheduler.state_dict()


def test_lr_transition_rejects_incorrect_source_before_loading_weights(tmp_path):
    parent, record = parent_checkpoint(tmp_path / "parent")
    old_lr = parent.settings["learning_rate"]
    phase = new_phase(tmp_path / "bad-lr", parent, record, lr=old_lr * 5,
                      transition={"from": old_lr * 2, "to": old_lr * 5})
    initial = weights(phase)
    with pytest.raises(ValueError, match="parent learning rate"):
        phase.start_from_parent(parent.run_dir)
    assert all(torch.equal(t, phase.model.state_dict()[name]) for name, t in initial.items())
