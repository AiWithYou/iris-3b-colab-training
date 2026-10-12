import copy
import random
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import torch

import iris_lora_colab as core
from iris_full_ft_phase_colab import state_sample_digest
from iris_preview_colab import preview_once


PROMPTS = [dict(name="test", prompt="test", seed=42, width=8, height=8)]


class Encoder:
    calls = 3
    overflow_rows = 0
    overflow_tokens = 0
    max_caption_tokens = 4
    device = "cpu"

    def to(self, device):
        self.device = str(device)
        return self


def trainer():
    model = torch.nn.Linear(3, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model(torch.ones(1, 3)).sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
    return SimpleNamespace(step=1100, model=model, optimizer=optimizer,
        scheduler=scheduler, encoder=Encoder(), device=torch.device("cpu"),
        cfg=SimpleNamespace(flow=SimpleNamespace(num_train_timesteps=1000, prediction="v")),
        _autocast=nullcontext)


def snapshot(value):
    return (state_sample_digest(core.rng_state()),
            state_sample_digest(value.model.state_dict()),
            state_sample_digest(value.optimizer.state_dict()),
            copy.deepcopy(value.scheduler.state_dict()))


def fake_generate(model, encoder, prompts, **kwargs):
    random.random()
    np.random.rand()
    encoder.calls += 1
    return torch.rand(1, 3, kwargs["height"], kwargs["width"]) * 2 - 1


def test_preview_preserves_rng_optimizer_weights_and_training_mode(tmp_path):
    value = trainer()
    before = snapshot(value)
    receipt = preview_once(value, tmp_path / "run", tmp_path / "scratch", PROMPTS,
                           target_step=1100, generate_fn=fake_generate)
    assert receipt["status"] == "preview_complete"
    assert len(receipt["images"]) == 1 and receipt["contact_sheet"]
    assert snapshot(value) == before
    assert value.model.training and value.encoder.device == "cpu" and value.encoder.calls == 3
    # The next stochastic optimizer update uses exactly the original random stream.
    rng = core.rng_state()
    actual = torch.rand(3)
    core.restore_rng(rng)
    assert torch.equal(torch.rand(3), actual)


def test_failed_preview_restores_state_and_records_failure(tmp_path):
    value = trainer()
    before = snapshot(value)

    def fail(*args, **kwargs):
        fake_generate(*args, **kwargs)
        raise RuntimeError("sampling failed")

    receipt = preview_once(value, tmp_path / "run", tmp_path / "scratch", PROMPTS, target_step=1100, generate_fn=fail)
    assert receipt["status"] == "preview_failed" and receipt["error"]["message"] == "sampling failed"
    assert snapshot(value) == before
    assert value.model.training and value.encoder.device == "cpu" and value.encoder.calls == 3


def test_existing_receipt_prevents_repeat_gpu_generation(tmp_path):
    value = trainer()
    first = preview_once(value, tmp_path / "run", tmp_path / "scratch", PROMPTS, target_step=1100, generate_fn=fake_generate)

    def unexpected(*args, **kwargs):
        raise AssertionError("A saved preview must not be generated again.")

    second = preview_once(value, tmp_path / "run", tmp_path / "scratch", PROMPTS, target_step=1100, generate_fn=unexpected)
    assert second == first


def test_preview_does_not_run_at_other_optimizer_steps(tmp_path):
    value = trainer()
    value.step = 1099
    assert preview_once(value, tmp_path / "run", tmp_path / "scratch", PROMPTS,
                        target_step=1100, generate_fn=fake_generate) is None
    assert not (tmp_path / "run").exists()


def test_configured_step_and_resolution_shift_are_used_in_image_metadata(tmp_path):
    value = trainer()
    value.step = 7
    value.phase_start_step = 2
    value.cfg.flow.shift = 3.0
    observed = []

    def generate(*args, **kwargs):
        observed.append(kwargs["shift"])
        return fake_generate(*args, **kwargs)

    receipt = preview_once(value, tmp_path / "run", tmp_path / "scratch", PROMPTS,
                           target_step=7, generate_fn=generate)
    assert receipt["optimizer_step"] == 7 and receipt["added_optimizer_steps"] == 5
    assert receipt["images"][0]["optimizer_step"] == 7
    assert receipt["images"][0]["added_optimizer_steps"] == 5
    assert observed == [3.0] and receipt["images"][0]["shift"] == 3.0
    assert (tmp_path / "run/previews/step-00000007/manifest.json").is_file()
