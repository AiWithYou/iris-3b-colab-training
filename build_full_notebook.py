"""Build a standalone Colab full-fine-tuning notebook, with common data cells."""
import json

from build_notebook import ROOT, build as build_lora, cell, with_dataset_config


def build():
    base = json.loads((ROOT / "Iris3B_LoRA_A100.ipynb").read_text(encoding="utf-8"))
    cells = []

    def md(source):
        cells.append(cell("markdown", source))

    def code(source):
        cells.append(cell("code", source))

    def common_code(prefix):
        found = ["".join(item["source"]) for item in base["cells"]
                 if item["cell_type"] == "code" and "".join(item["source"]).startswith(prefix)]
        if len(found) != 1:
            raise ValueError(f"Common notebook cell missing or duplicated: {prefix}")
        return found[0]

    md("""# Iris-3B 全体FT · A100 80GB / 1024px

A100 GPU・高RAMを選び、学習設定セルに自分のHugging Faceデータセット・固定commit SHA・列名を入力して、上から実行します。
Iris全パラメータをFP32で更新し、Qwenは凍結します。AdamW8bit、BF16 autocast、勾配チェックポイントを使います。

初期値は5,000枚、1024px相当の縦横比バケット、実バッチ14×累積2、学習率1e-5、300 optimizer更新です。
バッチ14はA100 80GB向けです。途中保存は100更新または30分、1回の実行は最大180分です。
再開には全重み・optimizer・scheduler・乱数・データ位置を保存します。完成モデルは公式形式のBF16です。

GPU・Drive容量を使います。再開用チェックポイントは1世代約18GB、生成用モデルは約6GBです。
Driveは初期学習用に約200GBの空きを推奨します。実行時間の設定は料金の上限ではありません。
公開版を任意のデータセットでA100に通した検証は未実施です。最初の更新・保存セルで確認してください。

[Iris公式コード](https://github.com/speridlabs/iris-3b) · [ベースモデル](https://huggingface.co/speridlabs/iris-3b) · [Colab FAQ](https://research.google.com/colaboratory/faq.html)
""")

    md("""## 1. 環境を準備

Python 3.11以上が必要です。CUDA版PyTorchは保持します。再起動を求められた場合はセッションを再起動し、このセルから再実行してください。
""")
    environment = common_code("#@title 依存関係と固定revision")
    environment = environment.replace("pins = {", 'pins = {\n    "bitsandbytes": ("0.49.2", "bitsandbytes"),', 1)
    environment = environment.replace("import importlib.metadata as metadata",
        'import os\nos.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"\nimport importlib.metadata as metadata', 1)
    code(environment)
    md("""## 2. 学習コード

画像の準備、保存・再開の共通コードと、全体FT用コードをColabへ保存します。
""")
    code("%%writefile /content/iris_lora_colab.py\n" + (ROOT / "iris_lora_colab.py").read_text(encoding="utf-8"))
    code("%%writefile /content/iris_full_ft_colab.py\n" + (ROOT / "iris_full_ft_colab.py").read_text(encoding="utf-8"))
    md("""## 3. Google Driveと学習設定

保存先は `MyDrive/Iris3B_FullFT/<RUN_NAME>/` です。解像度・学習率・seed・画像数・実バッチ・累積数を変更する場合はRUN_NAMEを変えてください。目標ステップ数、時間上限、保存間隔は再開後も変更できます。同じRUN_NAMEの学習を同時に複数のランタイムで動かさないでください。

1024pxではflow shift 4、512pxではshift 3を使います。5,000枚でまず適応と過学習を確認し、必要なら新しいRUN_NAMEで画像数を10,000枚へ増やせます。
""")
    code(with_dataset_config('''#@title 全体FTの設定
RUN_NAME = "iris3b-fullft-1024" #@param {type:"string"}
SAMPLE_COUNT = 5000 #@param {type:"integer"}
SEED = 42 #@param {type:"integer"}
RESOLUTION = 1024 #@param [512, 1024] {type:"raw"}
MIN_AESTHETIC_TIER = 0 #@param {type:"integer"}
ASPECT_BUCKETS = True #@param {type:"boolean"}
SFW_ONLY = False #@param {type:"boolean"}
LEARNING_RATE = 0.00001 #@param {type:"number"}
MICRO_BATCH_SIZE = 14 #@param {type:"integer"}
GRAD_ACCUM = 2 #@param {type:"integer"}
WARMUP_STEPS = 30 #@param {type:"integer"}
MAX_STEPS = 300 #@param {type:"integer"}
SESSION_MINUTES = 180 #@param {type:"integer"}
SAVE_EVERY = 100 #@param {type:"integer"}
SAVE_MINUTES = 30 #@param {type:"integer"}

import gc
import importlib
import json
import re
import shutil
import torch
from google.colab import drive
sys.path.insert(0, "/content")
import iris_lora_colab as iris
import iris_full_ft_colab as fullft
iris = importlib.reload(iris)
fullft = importlib.reload(fullft)

if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", RUN_NAME):
    raise ValueError("RUN_NAMEは英数字・ハイフン・アンダースコアの1〜80文字にしてください。")
if min(SAMPLE_COUNT, MICRO_BATCH_SIZE, GRAD_ACCUM, MAX_STEPS, SESSION_MINUTES, SAVE_EVERY, SAVE_MINUTES) < 1:
    raise ValueError("枚数、ステップ数、時間、保存間隔は1以上にしてください。")
if not 0 <= SEED < 2**32 or WARMUP_STEPS < 0 or LEARNING_RATE <= 0:
    raise ValueError("seed、warmup、学習率を確認してください。")
iris.flow_shift_for_resolution(RESOLUTION)
iris.check_a100()
if MICRO_BATCH_SIZE > 1 and torch.cuda.get_device_properties(0).total_memory < 75 * 1024**3:
    raise RuntimeError("この実バッチ設定はA100 80GB用です。40GBではMICRO_BATCH_SIZE=1から確認してください。")
memory = {line.split(":", 1)[0]: int(line.split()[1]) * 1024
          for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line}
if memory["MemTotal"] < 32 * 1024**3:
    raise RuntimeError("全体FTの保存用に高RAM（32GiB以上）のColabランタイムを選んでください。")

if not Path("/content/drive/MyDrive").is_dir():
    drive.mount("/content/drive", timeout_ms=600000)
DRIVE_ROOT = Path("/content/drive/MyDrive/Iris3B_FullFT")
RUN_DIR = DRIVE_ROOT / RUN_NAME
SCRATCH = Path("/content/iris-fullft-work") / RUN_NAME
RUN_DIR.mkdir(parents=True, exist_ok=True)
SCRATCH.mkdir(parents=True, exist_ok=True)
if shutil.disk_usage("/content").free < 60 * 1024**3 and "trainer" not in globals():
    raise RuntimeError("モデル取得・大きな途中保存用にColab側の空き60GiB以上を用意してください。")
settings = dict(sample_count=SAMPLE_COUNT, seed=SEED, resolution=RESOLUTION,
    aspect_buckets=ASPECT_BUCKETS, min_aesthetic_tier=MIN_AESTHETIC_TIER, sfw_only=SFW_ONLY,
    learning_rate=LEARNING_RATE, micro_batch_size=MICRO_BATCH_SIZE, grad_accum=GRAD_ACCUM, warmup_steps=WARMUP_STEPS,
    text_dropout=0.1, max_steps=MAX_STEPS, session_minutes=SESSION_MINUTES,
    save_every=SAVE_EVERY, save_minutes=SAVE_MINUTES, keep_checkpoints=2)
print("保存先:", RUN_DIR)
print("全体FT /", RESOLUTION, "px / 実バッチ", MICRO_BATCH_SIZE, "/ 累積", GRAD_ACCUM,
      "/ 実効バッチ", MICRO_BATCH_SIZE * GRAD_ACCUM, "/ 学習率", LEARNING_RATE)
print("Google Driveの空き目安: 約200GB（データ、複数世代の保存、最終モデル、Driveのごみ箱を含む）")
'''))
    md("""## 4. 学習画像を準備

指定した固定revisionのParquetから、設定した枚数・品質条件で画像を選びます。
重複ID、読めない画像、空captionを除外し、元画像と選択結果をDriveへ保存します。
公式の縦横比バケットへリサイズ・中央クロップし、入力captionをQwenに渡します。
必要ならSFW_ONLYを使います。ratingタグのない自然文captionでは無効のまま使ってください。
""")

    code(common_code("#@title 抽出または保存済みデータ"))
    code(common_code("#@title 抽出画像を確認"))
    md("""## 5. 全体FTモデルを読み込み、自動再開

Iris本体の全パラメータをFP32で読み込み、attention、MLP、pixel head、norm、共有modulation、テキストadapter等を全て更新します。Qwenは凍結します。学習中の逆伝播にはQwenをGPUに残しません。

最新の正常な保存を自動検出し、モデル全重み・8bit optimizer・scheduler・データ位置・CPU/GPUの乱数を復元します。最新ファイルの破損時は前の世代へ戻ります。設定違いや全保存の破損時は停止します。再開には約18GBのファイル転送が必要です。
""")
    code('''#@title Iris全体とoptimizerを読み込み、自動再開
for old_name in ("trainer", "model", "encoder"):
    globals().pop(old_name, None)
gc.collect()
torch.cuda.empty_cache()
model, encoder, cfg, targets = fullft.load_full_models(settings, Path("/content/iris-model-cache"))
trainer = fullft.FullFTTrainer(model, encoder, cfg, targets, dataset, settings,
                              RUN_DIR, SCRATCH, device="cuda")
trainer.resume()
estimate = fullft.storage_and_memory_estimate(model)
print("全パラメータを学習:", all(p.requires_grad for p in model.parameters()))
print("1周の使用枚数:", trainer.bucket_order.epoch_size, "/ バケット末尾の補充:", trainer.bucket_order.padding)
print("目標の周回数:", round(MAX_STEPS * MICRO_BATCH_SIZE * GRAD_ACCUM / trainer.bucket_order.epoch_size, 2))
print("重み+勾配+optimizerの概算:", round(estimate["persistent_training_tensor_gib"], 2), "GiB")
print("上記にactivation・一時テンソル・CUDAの消費が加わります。ピークは次の実機確認で測ります。")
print("再開用の保存1世代:", round(estimate["checkpoint_estimated_gib"], 2), "GiB（概算）")
''')
    md("""## 6. 学習前の比較作例

未学習のベースモデルから比較作例を保存します。再開時も比較画像がなければ固定revisionのベースから生成し、同じプロンプト・seed・stepsで学習後と比較します。学習用の乱数と画像位置は維持し、同条件の比較画像が保存済みなら再利用します。
""")
    code('''#@title 比較条件と学習前の作例
VALIDATION_PROMPTS = [
    "An illustration of a flower in a pot.",
    "An illustration of a lighthouse on a quiet coast.",
]
SAMPLE_STEPS = 50 #@param {type:"integer"}
SAMPLE_SEED = 42 #@param {type:"integer"}
baseline_conditions = dict(signature=trainer.signature, prompts=VALIDATION_PROMPTS,
    seed=SAMPLE_SEED, steps=SAMPLE_STEPS, resolution=RESOLUTION, cfg_scale=3.0, shift=trainer.flow.shift)
baseline_metadata = RUN_DIR / "samples/baseline.json"
baseline_paths = [RUN_DIR / "samples" / f"step-00000000-{i:02d}.png" for i in range(len(VALIDATION_PROMPTS))]
baseline_ready = (baseline_metadata.is_file() and all(p.is_file() for p in baseline_paths)
                  and json.loads(baseline_metadata.read_text()) == baseline_conditions)
if not baseline_ready:
    if trainer.step == 0:
        trainer.sample(VALIDATION_PROMPTS, steps=SAMPLE_STEPS, seed=SAMPLE_SEED)
    else:
        # Reproduce the untrained base even when the first update is already saved.
        from iris3b.models import IrisDiT
        from iris3b.sampling import generate, load_for_inference
        from PIL import Image
        reference_rng = iris.rng_state()
        comparison_model = None
        try:
            _, base_weights = load_for_inference(Path("/content/iris-model-cache/iris-base"))
            with torch.device("meta"):
                comparison_model = IrisDiT(trainer.cfg.model)
            comparison_model.load_state_dict(base_weights, strict=True, assign=True)
            del base_weights
            comparison_model.to(device="cuda", dtype=torch.float32).eval().requires_grad_(False)
            trainer.encoder.to("cuda")
            with torch.no_grad():
                for i, prompt in enumerate(VALIDATION_PROMPTS):
                    with trainer._autocast():
                        output = generate(comparison_model, trainer.encoder, [prompt],
                            height=RESOLUTION, width=RESOLUTION, steps=SAMPLE_STEPS,
                            cfg_scale=3.0, shift=trainer.flow.shift,
                            generator=torch.Generator(device="cuda").manual_seed(SAMPLE_SEED + i),
                            device="cuda", num_train_timesteps=trainer.cfg.flow.num_train_timesteps,
                            prediction=trainer.cfg.flow.prediction)[0]
                    pixels = output.detach().float().cpu().add(1).mul(127.5).clamp(0, 255)
                    image = Image.fromarray(pixels.permute(1, 2, 0).round().to(torch.uint8).numpy())
                    local_image = SCRATCH / baseline_paths[i].name
                    image.save(local_image)
                    iris.verified_copy(local_image, baseline_paths[i])
                    del output, pixels, image
                    print("ベース比較画像:", i + 1, "/", len(VALIDATION_PROMPTS), flush=True)
        finally:
            comparison_model = None
            trainer.encoder.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()
            iris.restore_rng(reference_rng)
    iris.atomic_json(baseline_metadata, baseline_conditions)
    baseline_ready = True
print("学習前の作例:", "保存済み" if baseline_ready else "保存なし（学習後のみ表示します）")
''')
    md("""## 7. A100実機で最小確認

初回は実データで1 optimizerステップ（実バッチ14 × 累積2）を実行し、全重みとoptimizerをDriveへ保存します。この更新は本学習の最初の1ステップです。初回の保存にも約18GBの容量と転送時間を使います。再開時は読み込んだoptimizer状態を含めてforward/backwardを確認します。
""")
    code('''#@title 最初の更新・保存と実際のGPUピークを確認
torch.cuda.reset_peak_memory_stats()
if trainer.step == 0:
    trainer.train(max_steps=1)
    if trainer.step < 1 or trainer.invalid_optimizer_state:
        raise RuntimeError("最初の更新が完了していません。必要なら読み込みセルから再実行してください。")
else:
    trainer.sanity_check()
print("確認完了 / step:", trainer.step)
print("PyTorch peak allocated:", round(torch.cuda.max_memory_allocated()/1024**3, 2), "GiB")
print("PyTorch peak reserved:", round(torch.cuda.max_memory_reserved()/1024**3, 2), "GiB")
''')
    md("""## 8. 本学習・続行

100 optimizerステップまたは30分ごと、時間上限到達時、正常終了時に全体を保存します。このセルの再実行で続けられます。初期値の実効バッチ28・300ステップは5,000枚を約1.6周する量です。バケット末尾の補充を含む正確な周回数はモデル読み込み時に表示します。目標を増やす前に作例の改善と過学習を確認してください。

通常の停止では途中の勾配累積を破棄し、完了した更新まで保存します。optimizer更新中の停止とランタイム強制終了は、前回の正常な保存から再開します。
""")
    code('''#@title 全体FTを開始・続行
trainer.update_runtime_settings(settings)
trainer.train()
''')
    md("""## 9. 学習前後を比較

学習前の保存が比較条件と一致すると、左がベース、右が全体FTです。保存が無い場合は学習後のみ表示します。画像は選択した解像度でDriveへ保存し、画面表示だけ最大512pxへ縮小します。
""")
    code('''#@title 同じ条件で学習後の作例を生成
from IPython.display import display
from PIL import Image, ImageDraw
paths = trainer.sample(VALIDATION_PROMPTS, steps=SAMPLE_STEPS, seed=SAMPLE_SEED)
baseline_ready = (baseline_metadata.is_file() and all(p.is_file() for p in baseline_paths)
                  and json.loads(baseline_metadata.read_text()) == baseline_conditions)
preview_size = min(512, trainer.settings["resolution"])
row_height = preview_size + 28
columns = 2 if baseline_ready else 1
sheet = Image.new("RGB", (columns * preview_size, row_height * len(paths)), "white")
draw = ImageDraw.Draw(sheet)
for i, path in enumerate(paths):
    entries = [(baseline_paths[i], "Base"), (path, f"Full FT step {trainer.step}")] if baseline_ready else [(path, f"Full FT step {trainer.step}")]
    for column, (image_path, label) in enumerate(entries):
        with Image.open(image_path) as image:
            image.thumbnail((preview_size, preview_size))
            sheet.paste(image, (column * preview_size, i * row_height + 28))
        draw.text((column * preview_size + 8, i * row_height + 8), label, fill="black")
display(sheet)
''')
    md("""## 10. モデル全体を書き出す

`export/step-xxxxxxxx/model.safetensors` と `config.yaml` を保存します。公式Irisで読み込めるBF16の全体モデル（約6GB）です。途中再開にはFP32の全重みとoptimizerが入った `checkpoints` を使います。書き出しには追加の保存容量と転送時間が必要です。
""")
    code('''#@title 学習済みの全体モデルをDriveへ保存
export_dir = fullft.export_full_model(trainer, RUN_DIR / "export")
for filename in ("iris_lora_colab.py", "iris_full_ft_colab.py"):
    iris.verified_copy(Path("/content") / filename, export_dir / filename)
print("保存済みoptimizerステップ:", trainer.step)
print("公式Iris形式のモデル:", export_dir)
''')
    md("""## 11. 切断前にDriveへの送信を完了

学習セルが停止した後、GPUランタイムを接続解除する前に実行します。Driveへのコピー後もアップロードが続く場合があるため、未送信データの同期完了まで待ちます。完了後にDriveを再び使う場合は、設定セルからマウントし直してください。
""")
    code('''#@title Driveへの送信を完了し、切断できる状態にする
from google.colab import drive
print("Google Driveへの未送信データを同期しています。", flush=True)
drive.flush_and_unmount(timeout_ms=1800000)
print("Driveへの送信完了。GPUランタイムを接続解除できます。", flush=True)
''')
    name = "Iris3B_FullFT_A100.ipynb"
    notebook = dict(nbformat=4, nbformat_minor=5, cells=cells,
        metadata=dict(colab=dict(name=name, provenance=[], gpuType="A100"), accelerator="GPU",
            kernelspec=dict(display_name="Python 3", language="python", name="python3"),
            language_info=dict(name="python", version="3.11")))
    target = ROOT / name
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {target.name}: {len(cells)} cells, {target.stat().st_size:,} bytes")
    return target


if __name__ == "__main__":
    build_lora()
    build_lora(resolution=512, rank=16, filename="Iris3B_LoRA_A100_512.ipynb")
    build()
