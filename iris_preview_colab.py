"""A configurable one-time preview without changing the training trajectory."""
from __future__ import annotations

import copy
import gc
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw

import iris_lora_colab as core
from iris_full_ft_phase_colab import state_sample_digest

DEFAULT_PROMPTS = [
    dict(name="sample", prompt="An illustration of a flower in a pot.", seed=42, width=1024, height=1024),
]


def preview_once(trainer, run_dir, scratch, prompts, *, target_step=300, generate_fn=None):
    """Save once at target_step; preserve RNG, optimizer, weights and mode."""
    if not isinstance(target_step, int) or target_step < 1:
        raise ValueError("target_step must be a positive integer.")
    if trainer.step != target_step:
        return None
    run_dir, scratch = Path(run_dir), Path(scratch)
    output_dir = run_dir / f"previews/step-{target_step:08d}"
    local_dir = scratch / f"previews/step-{target_step:08d}"
    shift = float(getattr(trainer.cfg.flow, "shift", 4.0))
    policy = dict(optimizer_step=target_step,
                  added_optimizer_steps=target_step - getattr(trainer, "phase_start_step", 0), prompts=prompts,
                  sampling_steps=50, cfg_scale=3.0, shift=shift)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        import json
        saved = json.loads(manifest_path.read_text())
        if all(saved.get(key) == value for key, value in policy.items()):
            if all(core.file_hash(run_dir / item["file"]) == item["sha256"]
                   for item in saved["images"]):
                return saved
        raise RuntimeError("The existing preview has a different identity or hash.")
    if generate_fn is None:
        from iris3b.sampling import generate
        generate_fn = generate
    output_dir.mkdir(parents=True, exist_ok=True)
    local_dir.mkdir(parents=True, exist_ok=True)
    rng = core.rng_state()
    rng_digest = state_sample_digest(rng)
    model_digest = state_sample_digest(trainer.model.state_dict())
    optimizer_digest = state_sample_digest(trainer.optimizer.state_dict())
    scheduler = copy.deepcopy(trainer.scheduler.state_dict())
    was_training = trainer.model.training
    stats = {name: getattr(trainer.encoder, name) for name in
             ("calls", "overflow_rows", "overflow_tokens", "max_caption_tokens")
             if hasattr(trainer.encoder, name)}
    images, contact_sheet, error = [], None, None
    started = time.monotonic()
    try:
        assert not any(parameter.grad is not None for parameter in trainer.model.parameters())
        trainer.model.eval()
        trainer.encoder.to(trainer.device)
        for item in prompts:
            with torch.no_grad(), trainer._autocast():
                output = generate_fn(trainer.model, trainer.encoder, [item["prompt"]],
                    height=item["height"], width=item["width"], steps=50,
                    cfg_scale=3.0, shift=shift,
                    generator=torch.Generator(device=trainer.device).manual_seed(item["seed"]),
                    device=trainer.device, num_train_timesteps=trainer.cfg.flow.num_train_timesteps,
                    prediction=trainer.cfg.flow.prediction)[0]
            assert output.shape == (3, item["height"], item["width"])
            assert torch.isfinite(output).all().item()
            array = output.detach().float().cpu().add(1).mul(127.5).clamp(0, 255)
            array = array.permute(1, 2, 0).round().to(torch.uint8).numpy()
            assert float(array.std()) > 1
            local_image = local_dir / (item["name"] + ".png")
            Image.fromarray(array).save(local_image)
            target = output_dir / local_image.name
            checksum = core.verified_copy(local_image, target)
            images.append(dict(file=str(target.relative_to(run_dir)), sha256=checksum,
                optimizer_step=target_step, added_optimizer_steps=policy["added_optimizer_steps"], **item,
                sampling_steps=50, cfg_scale=3.0, shift=shift, pixel_std=float(array.std())))
            del output, array
            print(f"STEP_{target_step}_PREVIEW: {len(images)}/{len(prompts)} / {item['name']}", flush=True)
        sheet = Image.new("RGB", (1024, ((len(images) + 1) // 2) * 548), "white")
        draw = ImageDraw.Draw(sheet)
        for index, item in enumerate(images):
            x, y = (index % 2) * 512, (index // 2) * 548
            draw.text((x + 8, y + 8), f"Step {target_step} / +{policy['added_optimizer_steps']} / {item['name']}", fill="black")
            with Image.open(run_dir / item["file"]) as source:
                source.thumbnail((512, 512))
                sheet.paste(source, (x + (512 - source.width) // 2, y + 30))
        local_sheet = local_dir / "contact-sheet.png"
        sheet.save(local_sheet)
        target = output_dir / local_sheet.name
        checksum = core.verified_copy(local_sheet, target)
        contact_sheet = dict(file=str(target.relative_to(run_dir)), sha256=checksum)
    except Exception as exc:
        error = dict(type=type(exc).__name__, message=str(exc))
        print(f"STEP_{target_step}_PREVIEW_FAILED: {error}; preserving training state and continuing.", flush=True)
    finally:
        try:
            trainer.encoder.to("cpu")
        finally:
            trainer.model.train(was_training)
            for name, value in stats.items():
                setattr(trainer.encoder, name, value)
            core.restore_rng(rng)
            gc.collect()
            if trainer.device.type == "cuda":
                torch.cuda.empty_cache()
    assert state_sample_digest(core.rng_state()) == rng_digest
    assert state_sample_digest(trainer.model.state_dict()) == model_digest
    assert state_sample_digest(trainer.optimizer.state_dict()) == optimizer_digest
    assert trainer.scheduler.state_dict() == scheduler
    receipt = dict(policy, status="preview_complete" if error is None else "preview_failed",
        images=images, contact_sheet=contact_sheet, error=error,
        training_rng_restored=True, model_samples_verified=True,
        optimizer_samples_verified=True, scheduler_verified=True,
        seconds=time.monotonic() - started)
    core.atomic_json(manifest_path, receipt)
    print(f"STEP_{target_step}_PREVIEW_SAVED: {len(images)} images; training state restored.", flush=True)
    return receipt
