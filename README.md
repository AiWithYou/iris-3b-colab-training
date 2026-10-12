# Iris-3B Colab Training

**日本語** | [English](README_EN.md)

Self-contained Colab notebooks for Iris-3B LoRA and full fine-tuning, with configurable datasets and resumable Google Drive checkpoints.

[Iris-3B公式](https://github.com/speridlabs/iris-3b)を使ったColab向け学習コードです。モデル構造、Qwenテキストエンコーダー、Rectified Flow、サンプラーは公式実装を使い、このリポジトリではLoRA、単一GPUでの全体FT、データ抽出、保存・再開、継続学習を提供します。

## Colabで開く

| 用途 | 初期設定 | ノートブック |
| --- | --- | --- |
| 全体FT | 1024px・実バッチ14×累積2・300更新 | [Open in Colab](https://colab.research.google.com/github/AiWithYou/iris-3b-colab-training/blob/main/Iris3B_FullFT_A100.ipynb) |
| 新しいデータで全体FTを継続 | optimizer・scheduler・乱数を引き継ぐ | [Open in Colab](https://colab.research.google.com/github/AiWithYou/iris-3b-colab-training/blob/main/Iris3B_FullFT_Continuation.ipynb) |
| LoRA | 1024px・rank 32・バッチ1×累積8 | [Open in Colab](https://colab.research.google.com/github/AiWithYou/iris-3b-colab-training/blob/main/Iris3B_LoRA_A100.ipynb) |
| LoRA 512px | 512px・rank 16・バッチ1×累積8 | [Open in Colab](https://colab.research.google.com/github/AiWithYou/iris-3b-colab-training/blob/main/Iris3B_LoRA_A100_512.ipynb) |

1. A100 GPUを選びます。全体FTの初期バッチ14はA100 80GB用で、高RAMも必要です。
2. 学習設定セルで、以下のデータ設定と自分の`RUN_NAME`を入力します。
3. 上から実行し、選択画像、最初の更新、保存を確認して学習を進めます。
4. 新しいランタイムでは同じ設定で再実行します。正常な最新チェックポイントを自動復元します。

全体FTの説明は[README_FULL_FT.md](README_FULL_FT.md)、新しいデータでの継続は[README_CONTINUATION.md](README_CONTINUATION.md)にあります。

## 学習データを指定する

Hugging Faceの任意のデータセットリポジトリにあるParquetファイルを読み込みます。データセットとrevisionの初期値は空欄です。画像フォルダーやCSVを直接読む機能は含みません。

| 設定 | 入力内容 |
| --- | --- |
| `DATASET_REPO` | 自分のデータセットの`owner/repository` |
| `DATASET_REVISION` | データセットの40桁commit SHA。`main`等の可変refは使いません |
| `PARQUET_GLOB` | 読むファイルのパターン。初期値`*.parquet`はサブフォルダー内も含みます |
| `ID_COLUMN` | 一意な画像IDの列。初期値`id` |
| `IMAGE_COLUMN` | 画像の列。初期値`image`。画像bytes、またはHF Image形式の`bytes`/`path`構造体に対応 |
| `PROMPT_COLUMN` | captionまたはタグ文字列の列。初期値`prompt` |
| `QUALITY_COLUMN` | 数値の品質区分の列。使わない場合は空欄 |
| `MIN_AESTHETIC_TIER` | 品質区分の下限。品質列を使わない場合は0 |

画像はParquet内に埋め込む必要があります。画像URLや、`bytes`が空で`path`だけの画像は読み込みません。PNG・JPEG・AVIF等、Pillowが読み込める画像に対応します。

空caption、重複ID、読めない画像を除外し、必要枚数を選びます。シャードの順序と各64行の読み込みバッチ内をseedでシャッフルする方式で、全データからの厳密な一様抽出ではありません。

`SFW_ONLY`はcaption内のratingタグで絞ります。自然文caption等、ratingタグのないデータではFalseのまま使います。データの利用条件と画像内容は各自のデータセットに従ってください。非公開・gatedデータが必要なら、Colab上で`huggingface_hub.login()`を使って認証します。

## 保存と再開

保存先は`MyDrive/Iris3B_LoRA/<RUN_NAME>/`または`MyDrive/Iris3B_FullFT/<RUN_NAME>/`です。選択データも保存し、同じ画像と順序を復元します。

チェックポイントには学習重み、optimizer、scheduler、optimizer更新数、データ位置、Python・NumPy・PyTorchの乱数状態を含みます。SHA-256を確認し、最新保存が壊れていれば前の世代へ戻ります。すべて壊れている場合は停止します。

データセット、revision、列名、解像度、学習率等を変更した同じrunからの再開は拒否します。目標更新数、実行時間上限、保存間隔は変更できます。同じrunを同時に複数のランタイムで実行しないでください。

全体FTの途中保存は1世代約18GB、BF16モデルは約6GBです。初期学習ではDriveに約200GBの空きを推奨します。既存の保存世代、データ、ごみ箱も容量を使います。実行時間の設定は料金の上限ではありません。

LoRAはこのコードのIris用形式です。一般的なSDXL/FLUX用ローダーとの互換性はありません。マージすると公式の`model.safetensors`と`config.yaml`形式で書き出せます。

## ソースと検証

| ファイル | 役割 |
| --- | --- |
| `iris_lora_colab.py` | LoRA、データ抽出、保存・再開、出力 |
| `iris_full_ft_colab.py` | Iris全パラメータ更新、実バッチ、AdamW8bit、公式形式の出力 |
| `iris_full_ft_phase_colab.py` | 新しいデータで学習状態を引き継ぐ |
| `iris_unseen_colab.py` | 以前の画像ID・SHA-256から除外リストを作る |
| `iris_preview_colab.py` | 指定した1更新時点だけ作例を保存し、学習状態を復元する |
| `build_*.py` | ソースを埋め込んだノートブックを生成する |
| `tests/` | 小型の実Iris構造と合成画像によるCPUテスト |

公式ソース、ベースモデル、Qwenのrevisionはコード内で固定しています。公開版はデータ設定とParquet列変換を含むCPUテストで検証します。任意のデータセットでのA100通し実行と、学習後の画質は未検証です。

Python 3.11以上で、ノートブックを再生成できます。

```sh
python build_full_notebook.py
python build_continuation_notebook.py
```

CPUテストの実行例です。Irisソースはダウンロードして参照し、モデル重みや学習画像はダウンロードしません。

```sh
python -m pip install "torch>=2.7.1,<2.14" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-dev.txt
git clone https://github.com/speridlabs/iris-3b.git vendor/iris-3b
git -C vendor/iris-3b checkout a8d15239dea469aba042cfa56ca3bb4e450d5ebc
```

Linux/macOS:

```sh
PYTHONPATH=vendor/iris-3b/src python -m pytest -q tests
```

PowerShell:

```powershell
$env:PYTHONPATH = (Resolve-Path vendor/iris-3b/src).Path
python -m pytest -q tests
```

## ライセンス

コードは[Apache License 2.0](LICENSE)です。公式Irisの著作権表記は[NOTICE](NOTICE)と[third_party/iris-NOTICE](third_party/iris-NOTICE)に保持しています。外部モデル、依存ライブラリ、利用者のデータセットにはそれぞれのライセンスが適用されます。
