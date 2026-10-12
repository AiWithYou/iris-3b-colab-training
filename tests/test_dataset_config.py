"""Verify user-selected data sources and their checkpoint identity."""
import io
import json
from pathlib import Path

import pytest
from PIL import Image

import iris_lora_colab as core
from test_iris_lora import make_trainer


def settings(**overrides):
    return dict(seed=42, dataset_repo="owner/images", dataset_revision="a" * 40,
                resolution=512, aspect_buckets=True, sample_count=2,
                min_aesthetic_tier=0, sfw_only=False, **overrides)


def mock_parquet(tmp_path, monkeypatch, rows):
    import huggingface_hub
    import pyarrow as pa
    import pyarrow.parquet as pq

    source = tmp_path / "fixture.parquet"
    pq.write_table(pa.Table.from_pylist(rows), source)
    downloaded = tmp_path / "copy.parquet"

    class Api:
        def list_repo_files(self, repo, **kwargs):
            assert repo == "owner/images" and kwargs["revision"] == "a" * 40
            return ["data/train-00000.parquet", "metadata.json"]

    def download(repo, name, **kwargs):
        assert repo == "owner/images" and kwargs["revision"] == "a" * 40
        assert name == "data/train-00000.parquet"
        downloaded.write_bytes(source.read_bytes())
        return str(downloaded)

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    return downloaded


def test_custom_columns_and_hf_image_struct_select_id_zero_without_quality(tmp_path, monkeypatch):
    rows = []
    for index in range(2):
        raw = io.BytesIO()
        Image.new("RGB", (24, 32), (index * 60, 20, 80)).save(raw, format="PNG")
        rows.append(dict(asset_id=index, picture=dict(bytes=raw.getvalue(), path=None),
                         description=f"illustration {index}"))
    copied = mock_parquet(tmp_path, monkeypatch, rows)
    configured = settings(dataset_columns=dict(image_id="asset_id", image="picture", prompt="description"))
    dataset = core.prepare_dataset(tmp_path / "drive", tmp_path / "scratch", configured)
    assert {row["image_id"] for row in dataset.records} == {"0", "1"}
    assert all(row["aesthetic_tier"] == 0 for row in dataset.records)
    assert dataset[0][0].shape[0] == 3 and not copied.exists()
    restored = core.prepare_dataset(tmp_path / "drive", tmp_path / "another", configured)
    assert restored.fingerprint == dataset.fingerprint


def test_missing_column_is_reported_and_download_removed(tmp_path, monkeypatch):
    copied = mock_parquet(tmp_path, monkeypatch, [dict(id="one", image=b"pixels", wrong="caption")])
    with pytest.raises(ValueError, match="Missing Parquet columns.*prompt"):
        next(core.dataset_rows(settings(), tmp_path))
    assert not copied.exists()


@pytest.mark.parametrize("change", [dict(dataset_revision="main"), dict(dataset_repo=""),
                                    dict(min_aesthetic_tier=1)])
def test_invalid_source_fails_before_any_network_call(tmp_path, monkeypatch, change):
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: pytest.fail("Unexpected network call"))
    configured = dict(settings(), **change)
    with pytest.raises(ValueError):
        next(core.dataset_rows(configured, tmp_path))


@pytest.mark.parametrize("field,value", [("dataset_repo", "owner/other"),
                                         ("dataset_revision", "b" * 40),
                                         ("dataset_columns", {"prompt": "caption"}),
                                         ("parquet_glob", "data/*.parquet")])
def test_dataset_configuration_changes_are_rejected_on_resume_and_live_update(tmp_path, field, value):
    parent = make_trainer(tmp_path)
    parent.train(max_steps=1)
    changed = make_trainer(tmp_path)
    changed.settings[field] = value
    changed.recipe = changed._build_recipe(changed.recipe["targets"])
    changed.signature = core.stable_hash(changed.recipe)
    with pytest.raises(ValueError, match="設定"):
        changed.resume()
    with pytest.raises(ValueError, match="設定"):
        parent.update_runtime_settings(dict(parent.settings, **{field: value}))


def test_continuation_notebook_is_standalone_and_has_empty_dataset_inputs():
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "Iris3B_FullFT_Continuation.ipynb").read_text(encoding="utf-8"))
    embedded = set()
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        assert cell["outputs"] == [] and cell["execution_count"] is None
        source = "".join(cell["source"])
        if source.startswith("%%writefile"):
            header, helper = source.split("\n", 1)
            name = header.rsplit("/", 1)[1]
            assert helper == (root / name).read_text(encoding="utf-8")
            embedded.add(name)
            compile(helper, name, "exec")
        else:
            compile(source, "continuation_cell", "exec")
    assert embedded == {"iris_lora_colab.py", "iris_full_ft_colab.py", "iris_full_ft_phase_colab.py",
                        "iris_unseen_colab.py", "iris_preview_colab.py"}
    configured = next("".join(cell["source"]) for cell in notebook["cells"]
                      if "".join(cell["source"]).startswith("#@title 全体FTの設定"))
    assert 'DATASET_REPO = ""' in configured and 'DATASET_REVISION = ""' in configured
    assert 'PARENT_RUN_NAME = ""' in configured


def test_continuation_loading_cell_restores_actual_parent_state_on_cpu(tmp_path):
    from types import SimpleNamespace
    import torch
    from iris_full_ft_phase_colab import PhaseFTTrainer
    from test_full_ft_phase import new_phase, parent_checkpoint

    parent, record = parent_checkpoint(tmp_path / "parent")
    template = new_phase(tmp_path / "phase", parent, record)
    configured = dict(template.settings, min_aesthetic_tier=0, dataset_repo="owner/other",
                      dataset_revision="b" * 40)
    for row in template.dataset.records:
        row["aesthetic_tier"] = 0

    def factory(*args, **kwargs):
        return PhaseFTTrainer(*args, **kwargs, device="cpu")

    scope = dict(Path=Path, fullft=SimpleNamespace(load_full_models=lambda *args: (
        template.model, template.encoder, template.cfg, template.recipe["targets"])),
        phaseft=SimpleNamespace(PhaseFTTrainer=factory), settings=configured,
        dataset=template.dataset, RUN_DIR=template.run_dir, SCRATCH=template.scratch,
        PARENT_RECORD=record, PARENT_DIR=parent.run_dir, transition=None)
    notebook = json.loads((Path(__file__).resolve().parents[1] /
                           "Iris3B_FullFT_Continuation.ipynb").read_text(encoding="utf-8"))
    source = next("".join(cell["source"]) for cell in notebook["cells"]
                  if "".join(cell["source"]).startswith("#@title 全体FTモデルを読み込み"))
    exec(compile(source, "continuation_loading_cell", "exec"), scope)
    restored = scope["trainer"]
    assert restored.step == parent.step and restored.cursor == parent.cursor
    assert all(torch.equal(tensor, restored.model.state_dict()[name])
               for name, tensor in parent.model.state_dict().items())
    assert restored.recipe["data"]["repo"] == "owner/other"
