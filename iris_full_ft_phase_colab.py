"""Continue full FT on a new dataset while retaining all training state."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch

import iris_lora_colab as core
from iris_full_ft_colab import FullFTTrainer


def state_sample_digest(value):
    """Compare every tensor's shape/dtype and boundary samples without a full GPU copy."""
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            digest.update(str((tuple(item.shape), str(item.dtype))).encode())
            flat = item.detach().reshape(-1)
            samples = torch.cat((flat[:16], flat[-16:])) if flat.numel() else flat
            digest.update(samples.cpu().contiguous().view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())

    visit(value)
    return digest.hexdigest()


class PhaseFTTrainer(FullFTTrainer):
    """Global optimizer/cursor accounting, with a new dataset starting at local zero."""

    def __init__(self, *args, parent_record, learning_rate_transition=None, **kwargs):
        self.parent_record = dict(parent_record)
        self.learning_rate_transition = dict(learning_rate_transition) if learning_rate_transition else None
        if self.learning_rate_transition and (set(self.learning_rate_transition) != {"from", "to"}
                or not all(isinstance(v, (int, float)) and v > 0
                           for v in self.learning_rate_transition.values())):
            raise ValueError("An explicit positive from/to learning-rate transition is required.")
        if (not core.CHECKPOINT_NAME.fullmatch(self.parent_record["file"])
                or self.parent_record["step"] < 1
                or len(self.parent_record["sha256"]) != 64):
            raise ValueError("Invalid parent checkpoint identity.")
        self.phase_start_step = self.parent_record["step"]
        self.phase_start_cursor = self.parent_record["cursor"]
        super().__init__(*args, **kwargs)
        if (self.learning_rate_transition
                and self.learning_rate_transition["to"] != self.settings["learning_rate"]):
            raise ValueError("The target learning rate must match the continuation settings.")
        effective_batch = self.settings["grad_accum"] * self.micro_batch_size
        if self.phase_start_cursor != self.phase_start_step * effective_batch:
            raise ValueError("Parent effective batch must match the continuation.")
        if any(row["aesthetic_tier"] < self.settings["min_aesthetic_tier"]
               for row in self.dataset.records):
            raise ValueError("Continuation images must meet the configured quality threshold.")

    def _build_recipe(self, targets):
        recipe = super()._build_recipe(targets)
        recipe["phase"] = dict(version="retain-state-new-dataset-v1",
            parent_checkpoint=self.parent_record, start_data_cursor=0)
        if self.learning_rate_transition:
            recipe["phase"].update(version="retain-state-new-dataset-lr-transition-v2",
                                  learning_rate_transition=self.learning_rate_transition)
        return recipe

    def index_at(self, cursor):
        local_cursor = cursor - self.phase_start_cursor
        if local_cursor < 0:
            raise ValueError("Load the parent checkpoint before drawing the new dataset.")
        return super().index_at(local_cursor)

    def start_from_parent(self, parent_run_dir):
        if (self.run_dir / "checkpoints/index.json").is_file():
            raise RuntimeError("The continuation already has a checkpoint; call resume().")
        parent_run_dir = Path(parent_run_dir)
        if parent_run_dir.resolve() == self.run_dir.resolve():
            raise ValueError("The parent and continuation must have separate save directories.")
        index = json.loads((parent_run_dir / "checkpoints/index.json").read_text())
        if index.get("schema") != core.SCHEMA_VERSION or self.parent_record not in index["checkpoints"]:
            raise RuntimeError("The exact verified parent checkpoint is absent.")
        source = parent_run_dir / "checkpoints" / self.parent_record["file"]
        local = self.scratch / ("phase-parent-" + self.parent_record["file"])
        core.verified_copy(source, local, self.parent_record["sha256"])
        try:
            payload = torch.load(local, map_location="cpu", weights_only=True, mmap=os.name != "nt")
            if (payload.get("schema") != core.SCHEMA_VERSION
                    or payload.get("step") != self.phase_start_step
                    or payload.get("cursor") != self.phase_start_cursor
                    or payload.get("signature") != core.stable_hash(payload["recipe"])):
                raise RuntimeError("Invalid parent schema, position, or recipe signature.")
            # Only data selection and its sampler length may change in this phase.
            ignore = {"data", "dataset_fingerprint", "sampler", "phase"}
            old = {k: v for k, v in payload["recipe"].items() if k not in ignore}
            current = {k: v for k, v in self.recipe.items() if k not in ignore}
            if self.learning_rate_transition:
                old = dict(old, training=dict(old["training"]))
                if old["training"]["learning_rate"] != self.learning_rate_transition["from"]:
                    raise ValueError("The parent learning rate does not match the authorized transition.")
                old["training"]["learning_rate"] = self.learning_rate_transition["to"]
            if old != current:
                raise ValueError("Architecture, optimizer, schedule, and training conditions must match the parent.")
            old_data = payload["recipe"]["data"]
            new_data = self.recipe["data"]
            data_ignore = {"sample_count", "min_aesthetic_tier", "selection", "exclusion",
                           "repo", "revision", "columns", "parquet_glob"}
            if ({k: v for k, v in old_data.items() if k not in data_ignore}
                    != {k: v for k, v in new_data.items() if k not in data_ignore}):
                raise ValueError("Resolution, preprocessing, seed and source revision must remain unchanged.")
            model_digest = state_sample_digest(payload["model"])
            optimizer_digest = state_sample_digest(payload["optimizer"])
            scheduler_state = payload["scheduler"]
            self._restore_weights(payload["model"])
            self.optimizer.load_state_dict(payload["optimizer"])
            self.scheduler.load_state_dict(scheduler_state)
            self.step, self.cursor = payload["step"], payload["cursor"]
            self.last_loss, self.committed_rng = payload["last_loss"], payload["rng"]
            self._epoch, self._order = None, None
            self.invalid_optimizer_state = False
            if state_sample_digest(self.model.state_dict()) != model_digest:
                raise RuntimeError("The parent model was not restored correctly.")
            if state_sample_digest(self.optimizer.state_dict()) != optimizer_digest:
                raise RuntimeError("The parent optimizer was not restored correctly.")
            if self.scheduler.state_dict() != scheduler_state:
                raise RuntimeError("The parent scheduler was not restored correctly.")
            moments_digest = state_sample_digest(self.optimizer.state_dict()["state"])
            if self.learning_rate_transition:
                if any(group["lr"] != self.learning_rate_transition["from"]
                       for group in self.optimizer.param_groups):
                    raise ValueError("The saved effective learning rate differs from the transition source.")
                new_lr = self.learning_rate_transition["to"]
                for group in self.optimizer.param_groups:
                    group["lr"] = group["initial_lr"] = new_lr
                updated_schedule = dict(scheduler_state,
                    base_lrs=[new_lr] * len(self.optimizer.param_groups),
                    _last_lr=[new_lr] * len(self.optimizer.param_groups))
                self.scheduler.load_state_dict(updated_schedule)
                if state_sample_digest(self.optimizer.state_dict()["state"]) != moments_digest:
                    raise RuntimeError("Changing the learning rate changed the optimizer moments.")
            core.restore_rng(self.committed_rng)
            if state_sample_digest(core.rng_state()) != state_sample_digest(self.committed_rng):
                raise RuntimeError("The parent random state was not restored correctly.")
            receipt = dict(parent=self.parent_record, retained_step=self.step,
                retained_global_cursor=self.cursor, new_dataset_cursor=0,
                parent_dataset_fingerprint=payload["recipe"]["dataset_fingerprint"],
                dataset_fingerprint=self.dataset.fingerprint,
                model_samples_verified=True, optimizer_samples_verified=True,
                scheduler_verified=True, rng_verified=True,
                model_sample_digest=model_digest, optimizer_sample_digest=optimizer_digest,
                learning_rate=self.optimizer.param_groups[0]["lr"],
                target_optimizer_step=self.settings["max_steps"])
            if self.learning_rate_transition:
                receipt.update(learning_rate_transition=self.learning_rate_transition,
                    optimizer_moments_retained=True,
                    optimizer_moments_digest=moments_digest,
                    retained_scheduler_epoch=self.scheduler.state_dict()["last_epoch"],
                    scheduler_base_lrs=self.scheduler.state_dict()["base_lrs"])
            core.atomic_json(self.run_dir / "phase_start.json", receipt)
        finally:
            local.unlink(missing_ok=True)
        print(f"Parent restored: step {self.step}, model + optimizer + scheduler + RNG. New dataset starts at 0.", flush=True)
        return receipt

    def save(self):
        if self.step <= self.phase_start_step:
            return None
        return super().save()
