"""Build the standalone Colab notebook without requiring nbformat."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def cell(kind, source):
    result = dict(cell_type=kind, id=hashlib.sha256(source.encode()).hexdigest()[:12],
                  metadata={}, source=source.splitlines(keepends=True))
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


DATASET_PARAMETERS = '''DATASET_REPO = "" #@param {type:"string"}
DATASET_REVISION = "" #@param {type:"string"}
PARQUET_GLOB = "*.parquet" #@param {type:"string"}
ID_COLUMN = "id" #@param {type:"string"}
IMAGE_COLUMN = "image" #@param {type:"string"}
PROMPT_COLUMN = "prompt" #@param {type:"string"}
QUALITY_COLUMN = "" #@param {type:"string"}
'''


def with_dataset_config(source):
    title, body = source.split("\n", 1)
    body = body.replace("settings = dict(sample_count=", "settings = dict(**dataset_settings, sample_count=", 1)
    validation = '''dataset_settings = dict(dataset_repo=DATASET_REPO, dataset_revision=DATASET_REVISION,
    parquet_glob=PARQUET_GLOB, dataset_columns=dict(image_id=ID_COLUMN, image=IMAGE_COLUMN,
    prompt=PROMPT_COLUMN, aesthetic_tier=QUALITY_COLUMN or None))
iris.dataset_config(dict(dataset_settings, min_aesthetic_tier=MIN_AESTHETIC_TIER), require_source=True)
'''
    body = body.replace("iris.flow_shift_for_resolution(RESOLUTION)", validation + "iris.flow_shift_for_resolution(RESOLUTION)", 1)
    return title + "\n" + DATASET_PARAMETERS + body


def build(resolution=1024, rank=32, filename="Iris3B_LoRA_A100.ipynb"):
    if resolution not in (512, 1024) or rank not in (8, 16, 32):
        raise ValueError("Unsupported notebook profile")
    helper = (ROOT / "iris_lora_colab.py").read_text(encoding="utf-8")
    cells = []
    def md(text):
        cells.append(cell("markdown", text))
    def code(text):
        cells.append(cell("code", text))

    md(f"""# Iris-3B LoRA学習 · A100 / {resolution}px

A100 GPUを選び、学習設定セルに自分のHugging Faceデータセット・固定commit SHA・列名を入力して、上から実行します。
ノートブック単体で動作します。Iris本体とQwenは固定revisionを使います。

初期値は5,000枚、{resolution}px相当の縦横比バケット、LoRA rank {rank} / alpha {rank}、バッチ1・勾配累積8です。
100更新または10分ごとにLoRA・optimizer・scheduler・乱数・データ位置をGoogle Driveへ保存します。
新しいランタイムでは同じRUN_NAMEと設定で再実行してください。解像度・データ・学習設定を変える場合はRUN_NAMEも変えます。

1回の実行は最大180分です。これは料金の上限ではありません。LoRAのColab通し実行は未検証です。

[Iris公式コード](https://github.com/speridlabs/iris-3b) · [ベースモデル](https://huggingface.co/speridlabs/iris-3b) · [Colab FAQ](https://research.google.com/colaboratory/faq.html)
""")

    md("""## 1. 環境を準備

Python 3.11以上が必要です。対応済みのCUDA版PyTorchは保持します。依存関係を更新して再起動を求められた場合は、ランタイムのセッションを再起動して、このセルから再実行してください。
""")
    code('''#@title 依存関係と固定revisionのIrisソースを準備
import importlib.metadata as metadata
import subprocess
import sys
from pathlib import Path
from packaging.version import Version

if sys.version_info < (3, 11):
    raise RuntimeError("Python 3.11以上のColabランタイムを選んでください。")
pins = {
    "transformers": ("5.10.4", "transformers"),
    "huggingface-hub": ("1.5.0", "huggingface_hub"),
    "safetensors": ("0.8.0", "safetensors"),
    "omegaconf": ("2.3.1", "omegaconf"),
    "pillow": ("12.3.0", "PIL"),
    "numpy": ("2.3.5", "numpy"),
    "pyarrow": ("26.0.0", "pyarrow"),
}
requirements, restart = [], []
for distribution, (wanted, module) in pins.items():
    try:
        installed = metadata.version(distribution)
    except metadata.PackageNotFoundError:
        installed = None
    if installed != wanted:
        requirements.append(f"{distribution}=={wanted}")
        if module in sys.modules:
            restart.append(module)
try:
    torch_version = metadata.version("torch")
except metadata.PackageNotFoundError:
    torch_version = "0"
if Version(torch_version.split("+")[0]) < Version("2.7.1"):
    requirements.append("torch>=2.7.1,<2.14")
    if "torch" in sys.modules:
        restart.append("torch")
if requirements:
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                    "--disable-pip-version-check", *requirements], check=True)
if restart:
    raise RuntimeError("依存関係を更新しました。ランタイムのセッションを再起動して、上から再実行してください。")

SOURCE_REVISION = "a8d15239dea469aba042cfa56ca3bb4e450d5ebc"
SOURCE_ROOT = Path("/content/iris-3b-source")
if not SOURCE_ROOT.exists():
    subprocess.run(["git", "clone", "--quiet", "--no-checkout",
                    "https://github.com/speridlabs/iris-3b.git", str(SOURCE_ROOT)], check=True)
    subprocess.run(["git", "-C", str(SOURCE_ROOT), "checkout", "--quiet", "--detach", SOURCE_REVISION], check=True)
actual = subprocess.check_output(["git", "-C", str(SOURCE_ROOT), "rev-parse", "HEAD"], text=True).strip()
if actual != SOURCE_REVISION:
    raise RuntimeError("Irisソースのrevisionが違います。新しいランタイムで上から実行してください。")
sys.path.insert(0, str(SOURCE_ROOT / "src"))
print("環境を準備しました。Iris revision:", actual[:12])
''')
    md("""## 2. 学習用コード

次のセルは、このノートブックに含まれる学習コードをColabへ保存します。
""")
    code("%%writefile /content/iris_lora_colab.py\n" + helper)
    md("""## 3. Google Driveと学習設定

保存先は `MyDrive/Iris3B_LoRA/<RUN_NAME>/`。初回はGoogle Driveの接続を許可します。rank・学習率・seed・枚数等を変更する場合はRUN_NAMEも変えてください。最大ステップ数、1回の時間上限、保存間隔は再開後も変更できます。

画像を切り抜きすぎないよう、初期値では公式の縦横比バケットを使用します。RESOLUTION=1024では正方形が1024×1024、縦長・横長も約1024²画素です。512にも切り替えられます。解像度を変更する場合もRUN_NAMEを変えてください。SFW_ONLYは安全側にタグを絞りますが、タグの誤分類までは除去できません。
""")
    code(with_dataset_config('''#@title 学習設定
RUN_NAME = "iris3b-lora-1024-r32" #@param {type:"string"}
SAMPLE_COUNT = 5000 #@param {type:"integer"}
SEED = 42 #@param {type:"integer"}
RESOLUTION = 1024 #@param [512, 1024] {type:"raw"}
MIN_AESTHETIC_TIER = 0 #@param {type:"integer"}
ASPECT_BUCKETS = True #@param {type:"boolean"}
SFW_ONLY = False #@param {type:"boolean"}
LORA_RANK = 32 #@param [8, 16, 32] {type:"raw"}
LORA_ALPHA = 32 #@param {type:"integer"}
LEARNING_RATE = 0.00005 #@param {type:"number"}
GRAD_ACCUM = 8 #@param {type:"integer"}
WARMUP_STEPS = 100 #@param {type:"integer"}
MAX_STEPS = 2000 #@param {type:"integer"}
SESSION_MINUTES = 180 #@param {type:"integer"}
SAVE_EVERY = 100 #@param {type:"integer"}
SAVE_MINUTES = 10 #@param {type:"integer"}

import gc
import importlib
import re
import shutil
import torch
from google.colab import drive
sys.path.insert(0, "/content")
import iris_lora_colab as iris

if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", RUN_NAME):
    raise ValueError("RUN_NAMEは英数字・ハイフン・アンダースコアの1〜80文字にしてください。")
if min(SAMPLE_COUNT, GRAD_ACCUM, MAX_STEPS, SESSION_MINUTES, SAVE_EVERY, SAVE_MINUTES) < 1:
    raise ValueError("枚数、ステップ数、時間、保存間隔は1以上にしてください。")
if not 0 <= SEED < 2**32 or WARMUP_STEPS < 0:
    raise ValueError("seedは0〜2**32-1、warmupは0以上にしてください。")
iris.flow_shift_for_resolution(RESOLUTION)
iris.check_a100()
drive.mount("/content/drive")
DRIVE_ROOT = Path("/content/drive/MyDrive/Iris3B_LoRA")
RUN_DIR = DRIVE_ROOT / RUN_NAME
SCRATCH = Path("/content/iris-work") / RUN_NAME
RUN_DIR.mkdir(parents=True, exist_ok=True)
SCRATCH.mkdir(parents=True, exist_ok=True)
if shutil.disk_usage("/content").free < 30 * 1024**3:
    raise RuntimeError("Colab側のローカルディスクに30GiB以上の空きを用意してください。")
settings = dict(sample_count=SAMPLE_COUNT, seed=SEED, resolution=RESOLUTION,
    aspect_buckets=ASPECT_BUCKETS, min_aesthetic_tier=MIN_AESTHETIC_TIER, sfw_only=SFW_ONLY,
    rank=LORA_RANK, alpha=LORA_ALPHA, learning_rate=LEARNING_RATE, grad_accum=GRAD_ACCUM,
    warmup_steps=WARMUP_STEPS, text_dropout=0.1, max_steps=MAX_STEPS,
    session_minutes=SESSION_MINUTES, save_every=SAVE_EVERY, save_minutes=SAVE_MINUTES,
    keep_checkpoints=3)
print("保存先:", RUN_DIR)
print("学習解像度:", RESOLUTION, "/ rank:", LORA_RANK, "/ flow shift:", iris.flow_shift_for_resolution(RESOLUTION))
print("バッチ1 / 累積", GRAD_ACCUM, "/ 最大", MAX_STEPS, "optimizerステップ")
'''.replace("iris3b-lora-1024-r32", f"iris3b-lora-{resolution}-r{rank}")
       .replace("RESOLUTION = 1024", f"RESOLUTION = {resolution}")
       .replace("LORA_RANK = 32", f"LORA_RANK = {rank}")
       .replace("LORA_ALPHA = 32", f"LORA_ALPHA = {rank}")))
    md("""## 4. 学習画像を準備

指定したParquetファイルの順序と各読み込みバッチ内をseedでシャッフルし、必要枚数を選びます。
全データからの厳密な一様抽出ではありません。読めない画像、重複ID、空captionを除外し、設定した品質条件を適用します。
選択結果と元画像をDriveへ保存し、再実行では同じデータを復元します。画像の内容と利用条件は、学習前に確認してください。
""")

    code('''#@title 抽出または保存済みデータを復元
dataset = iris.prepare_dataset(DRIVE_ROOT, SCRATCH, settings)
print("画像数:", len(dataset), "/ データ識別子:", dataset.fingerprint[:16])
''')
    code('''#@title 抽出画像を確認
from IPython.display import display
from PIL import Image
import numpy as np

tiles = []
for i in range(min(8, len(dataset))):
    pixels, caption = dataset[i]
    image = Image.fromarray(((pixels.permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8))
    image.thumbnail((192, 192))
    tile = Image.new("RGB", (192, 192), "white")
    tile.paste(image, ((192 - image.width) // 2, (192 - image.height) // 2))
    tiles.append(tile)
sheet = Image.new("RGB", (192 * 4, 192 * 2), "white")
for i, tile in enumerate(tiles):
    sheet.paste(tile, ((i % 4) * 192, (i // 4) * 192))
display(sheet)
''')
    md("""## 5. モデルを読み込み、保存済み学習を再開

IrisのベースとQwenは固定revisionを取得します。ベースはBF16で凍結し、LoRAのパラメータはFP32で学習します。Qwenは凍結したままGPUに置き、入力captionをIrisのtokenizer・テンプレートで処理します。データセット付属のCLIP用トークン数は使いません。

最新の保存が破損している場合は前の世代を試します。設定や画像が変わっていたり、全ての保存が破損していたりする場合は停止します。
""")
    code('''#@title モデルとLoRAを読み込み、自動再開
for old_name in ("trainer", "model", "encoder"):
    globals().pop(old_name, None)
gc.collect()
torch.cuda.empty_cache()
model, encoder, cfg, targets = iris.load_models(settings, Path("/content/iris-model-cache"))
trainer = iris.LoRATrainer(model, encoder, cfg, targets, dataset, settings,
                           RUN_DIR, SCRATCH, device="cuda")
trainer.resume()
''')
    code('''#@title 実際の学習画像でforward/backwardとメモリを確認
trainer.sanity_check()
print("確認用の勾配と乱数を元に戻しました。学習ステップは増えていません。")
''')
    md("""## 6. 学習

100 optimizerステップまたは10分ごとに保存します。1回の時間上限に達すると、最後のステップを保存して終了します。このセルを再実行すると続けられます。画像5,000枚・累積8では約625 optimizerステップが1周分です。

通常のセル停止は途中の勾配累積を破棄し、完了した更新まで保存します。optimizer更新の最中に停止した場合は読み込みセルから再実行してください。ランタイム強制終了の場合は、同じ設定で上から実行し、前回の正常な保存から続けます。
""")
    code('''#@title 学習開始・続行
# 時間・目標ステップ・保存間隔を変更した場合は、この直前に設定セルも実行してください。
trainer.update_runtime_settings(settings)
trainer.train()
''')
    md("""## 7. 同じseedでベースと学習後を比較

左側をベース、右側をLoRAとして確認します。比較は両方とも50 steps / CFG 3 / 設定した解像度です。表示は最大512pxへ縮小しますが、Driveには元の解像度の作例を保存します。学習の乱数状態を変えずに作例とプロンプト条件をDriveへ保存します。タグと自然文の両方を試せます。
""")
    code('''#@title 作例を生成
VALIDATION_PROMPTS = [
    "An illustration of a flower in a pot.",
    "An illustration of a lighthouse on a quiet coast.",
]
SAMPLE_STEPS = 50 #@param {type:"integer"}
SAMPLE_SEED = 42 #@param {type:"integer"}
paths = trainer.sample(VALIDATION_PROMPTS, steps=SAMPLE_STEPS, seed=SAMPLE_SEED, compare_base=True)
from PIL import Image, ImageDraw
preview_size = min(512, trainer.settings["resolution"])
row_height = preview_size + 28
sheet = Image.new("RGB", (2 * preview_size, row_height * len(VALIDATION_PROMPTS)), "white")
draw = ImageDraw.Draw(sheet)
for i in range(len(VALIDATION_PROMPTS)):
    for column, file_index in enumerate((i, i + len(VALIDATION_PROMPTS))):
        with Image.open(paths[file_index]) as image:
            image.thumbnail((preview_size, preview_size))
            sheet.paste(image, (column * preview_size, i * row_height + 28))
        draw.text((column * preview_size + 8, i * row_height + 8), "Base" if column == 0 else f"LoRA step {trainer.step}", fill="black")
display(sheet)
''')
    md("""## 8. LoRAを書き出す

`export/adapter.safetensors` と `adapter_config.json` をDriveへ保存します。これは本ノートブックのIris用LoRA形式で、一般的なSDXL/FLUXローダーにはそのまま読み込めません。読み込みに使う学習コードも一緒に保存します。

`EXPORT_MERGED=True` にすると、公式Irisで読み込める `merged/model.safetensors` + `config.yaml` も出力します。BF16なので約6GBの追加保存容量と、Colab側にも追加のCPUメモリが必要です。途中再開には `checkpoints` を使います。
""")
    code('''#@title LoRA保存（マージは任意）
EXPORT_MERGED = False #@param {type:"boolean"}
if trainer.step == 0 or trainer.invalid_optimizer_state:
    raise RuntimeError("正常な学習済みモデルを読み込んでから書き出してください。")
export_dir = iris.export_adapter(trainer.model, RUN_DIR / "export", trainer.recipe, trainer.step)
iris.verified_copy(Path("/content/iris_lora_colab.py"), export_dir / "iris_lora_colab.py")
if EXPORT_MERGED:
    iris.export_merged(trainer.model, trainer.cfg, RUN_DIR / "merged")
print("保存済みoptimizerステップ:", trainer.step)
print("出力先:", RUN_DIR)
''')
    notebook = dict(nbformat=4, nbformat_minor=5, cells=cells,
        metadata=dict(colab=dict(name=filename, provenance=[], gpuType="A100"),
            accelerator="GPU", kernelspec=dict(display_name="Python 3", language="python", name="python3"),
            language_info=dict(name="python", version="3.11")))
    target = ROOT / filename
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {target.name}: {len(cells)} cells, {target.stat().st_size:,} bytes")
    return target


if __name__ == "__main__":
    build()
    build(resolution=512, rank=16, filename="Iris3B_LoRA_A100_512.ipynb")
