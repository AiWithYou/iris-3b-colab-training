"""Previously selected IDs and identical image bytes cannot enter a new pool."""
import hashlib
import io

import pytest
from PIL import Image

import iris_lora_colab as core
from iris_unseen_colab import build_exclusion_ledger, exclusion_recipe


def png(color):
    stream = io.BytesIO()
    Image.new("RGB", (32, 40), color).save(stream, format="PNG")
    return stream.getvalue()


def old_manifest():
    records = [dict(image_id="old", image_sha256=hashlib.sha256(png("red")).hexdigest())]
    value = dict(recipe={"previous_pool": 1}, records=records)
    return dict(value, fingerprint=core.stable_hash(value))


def test_unseen_selection_filters_ids_hashes_tiers_and_restores_same_pool(tmp_path):
    ledger = build_exclusion_ledger([old_manifest()])
    settings = dict(seed=42, resolution=512, aspect_buckets=False, sample_count=2,
                    min_aesthetic_tier=3, sfw_only=False, exclusion=exclusion_recipe(ledger))
    rows = [dict(image_id=identity, image=png(color), aesthetic_tier=tier, prompt="masterpiece")
            for identity, color, tier in [("old", "green", 3), ("copied-old", "red", 3),
                ("low-tier", "blue", 2), ("new-green", "green", 3),
                ("copied-new", "green", 3), ("new-blue", "blue", 3)]]
    dataset = core.prepare_dataset(tmp_path / "drive", tmp_path / "scratch", settings,
                                   source_factory=lambda: iter(rows), exclusion=ledger)
    assert [row["image_id"] for row in dataset.records] == ["new-green", "new-blue"]
    restored = core.prepare_dataset(tmp_path / "drive", tmp_path / "other-scratch", settings,
                                    source_factory=lambda: iter(()), exclusion=ledger)
    assert restored.records == dataset.records and restored.fingerprint == dataset.fingerprint
    with pytest.raises(ValueError, match="exact prior-image"):
        core.prepare_dataset(tmp_path / "drive", tmp_path / "scratch", settings,
                             source_factory=lambda: iter(rows), exclusion=dict(ledger, image_ids=[]))


def test_exclusion_ledger_rejects_changed_previous_records():
    previous = old_manifest()
    previous["records"][0]["image_id"] = "changed"
    with pytest.raises(ValueError, match="fingerprint"):
        build_exclusion_ledger([previous])
