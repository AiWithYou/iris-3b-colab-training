"""Build a standalone full-FT continuation notebook with user-selected data."""
import json

from build_notebook import ROOT, cell


def build():
    base = json.loads((ROOT / "Iris3B_FullFT_A100.ipynb").read_text(encoding="utf-8"))
    cells = []

    def common(prefix):
        matches = ["".join(item["source"]) for item in base["cells"]
                   if item["cell_type"] == "code" and "".join(item["source"]).startswith(prefix)]
        if len(matches) != 1:
            raise ValueError(f"Missing or duplicated full-FT cell: {prefix}")
        return matches[0]

    def md(source):
        cells.append(cell("markdown", source))

    def code(source):
        cells.append(cell("code", source))

    md("""# Iris-3B 全体FT · 新しいデータで継続学習

この公開版で保存した全体FTチェックポイントから、重み・optimizer・scheduler・乱数を引き継ぎます。
新しいデータの位置は0から始め、optimizer更新数は通算で数えます。
先に親runを停止し、同時実行を避けてください。A100 80GB・高RAM・十分なDrive容量が必要です。

`PARENT_RUN_NAME`、別名の`RUN_NAME`、データセットと列名を設定します。seed・解像度・バッチ・warmup等は親と一致させます。
`MAX_STEPS`は通算の目標です。学習率を変える場合は、親の設定値を`PARENT_LEARNING_RATE`に明示します。
以前の画像を除外する場合は、Drive内のdataset manifestの相対パスを`EXCLUSION_MANIFESTS_JSON`にJSON配列で指定します。
除外には画像IDとSHA-256を使うため、画像IDはデータセット間でも一意になるよう用意してください。
""")
    code(common("#@title 依存関係と固定revision"))
    for name in ("iris_lora_colab.py", "iris_full_ft_colab.py", "iris_full_ft_phase_colab.py",
                 "iris_unseen_colab.py", "iris_preview_colab.py"):
        code(f"%%writefile /content/{name}\n" + (ROOT / name).read_text(encoding="utf-8"))

    settings = common("#@title 全体FTの設定")
    settings = settings.replace('RUN_NAME = "iris3b-fullft-1024"',
        'PARENT_RUN_NAME = "" #@param {type:"string"}\n'
        'PARENT_LEARNING_RATE = 0.00001 #@param {type:"number"}\n'
        'EXCLUSION_MANIFESTS_JSON = "[]" #@param {type:"string"}\n'
        'RUN_NAME = "iris3b-fullft-continuation"', 1)
    settings = settings.replace("MAX_STEPS = 300", "MAX_STEPS = 600", 1)
    settings = settings.replace('if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", RUN_NAME):',
        'if (not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", PARENT_RUN_NAME) or PARENT_RUN_NAME == RUN_NAME):\n'
        '    raise ValueError("親のPARENT_RUN_NAMEを入力し、RUN_NAMEには別の名前を指定してください。")\n'
        'if PARENT_LEARNING_RATE <= 0:\n'
        '    raise ValueError("PARENT_LEARNING_RATEは正の値にしてください。")\n'
        'if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", RUN_NAME):', 1)
    code(settings)
    code('''#@title 親チェックポイントと除外条件を確認
import iris_full_ft_phase_colab as phaseft
import iris_unseen_colab as unseen
phaseft = importlib.reload(phaseft)
unseen = importlib.reload(unseen)
PARENT_DIR = DRIVE_ROOT / PARENT_RUN_NAME
parent_index = json.loads((PARENT_DIR / "checkpoints/index.json").read_text(encoding="utf-8"))
if parent_index.get("schema") != iris.SCHEMA_VERSION or not parent_index.get("checkpoints"):
    raise RuntimeError("この公開版の正常な親チェックポイントが必要です。")
phase_receipt = RUN_DIR / "phase_start.json"
if phase_receipt.is_file():
    PARENT_RECORD = json.loads(phase_receipt.read_text(encoding="utf-8"))["parent"]
else:
    PARENT_RECORD = max(parent_index["checkpoints"], key=lambda item: item["step"])
if PARENT_RECORD not in parent_index["checkpoints"]:
    raise RuntimeError("最初に使った親チェックポイントを保存したまま再開してください。")
if MAX_STEPS <= PARENT_RECORD["step"]:
    raise ValueError("MAX_STEPSには親のstepより大きい通算目標を指定してください。")
paths = json.loads(EXCLUSION_MANIFESTS_JSON)
if not isinstance(paths, list) or any(not isinstance(path, str) for path in paths):
    raise ValueError("EXCLUSION_MANIFESTS_JSONはmanifest相対パスのJSON配列です。")
manifests = []
for relative in paths:
    path = (DRIVE_ROOT / relative).resolve()
    if not path.is_relative_to((DRIVE_ROOT / "datasets").resolve()) or path.name != "manifest.json":
        raise ValueError("datasets/配下のmanifest.jsonを指定してください。")
    manifests.append(json.loads(path.read_text(encoding="utf-8")))
ledger = unseen.build_exclusion_ledger(manifests) if manifests else None
if ledger is not None:
    settings["exclusion"] = unseen.exclusion_recipe(ledger)
transition = ({"from": PARENT_LEARNING_RATE, "to": LEARNING_RATE}
              if LEARNING_RATE != PARENT_LEARNING_RATE else None)
print("親step:", PARENT_RECORD["step"], "/ 通算目標:", MAX_STEPS)
''')
    code('''#@title 新しいデータを準備または復元
dataset = iris.prepare_dataset(DRIVE_ROOT, SCRATCH, settings, exclusion=ledger)
print("画像数:", len(dataset), "/ データ識別子:", dataset.fingerprint[:16])
''')
    code('''#@title 全体FTモデルを読み込み、親状態または途中保存を復元
model, encoder, cfg, targets = fullft.load_full_models(settings, Path("/content/iris-model-cache"))
trainer = phaseft.PhaseFTTrainer(model, encoder, cfg, targets, dataset, settings, RUN_DIR, SCRATCH,
    parent_record=PARENT_RECORD, learning_rate_transition=transition)
if (RUN_DIR / "checkpoints/index.json").is_file():
    trainer.resume()
else:
    trainer.start_from_parent(PARENT_DIR)
print("復元済みstep:", trainer.step, "/ 通算画像位置:", trainer.cursor)
''')
    code('''#@title 目標または時間上限まで学習して保存
trainer.update_runtime_settings(settings)
trainer.train(max_steps=MAX_STEPS)
print("保存済みstep:", trainer.step)
''')
    code('''#@title 公式形式のBF16モデルを書き出す
export_dir = fullft.export_full_model(trainer, RUN_DIR / "export" / f"step-{trainer.step:08d}")
print("モデルとconfigの保存先:", export_dir)
''')
    code(common("#@title Driveへの送信を完了"))
    name = "Iris3B_FullFT_Continuation.ipynb"
    notebook = dict(nbformat=4, nbformat_minor=5, cells=cells, metadata=base["metadata"])
    notebook["metadata"]["colab"]["name"] = name
    target = ROOT / name
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {name}: {len(cells)} cells, {target.stat().st_size:,} bytes")
    return target


if __name__ == "__main__":
    build()
