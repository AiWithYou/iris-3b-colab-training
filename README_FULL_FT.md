# Iris-3B 全体FT

**日本語** | [English](README_FULL_FT_EN.md)

[Colabで開く](https://colab.research.google.com/github/AiWithYou/iris-3b-colab-training/blob/main/Iris3B_FullFT_A100.ipynb)

Iris全パラメータをFP32で更新し、Qwenは凍結します。AdamW8bit、BF16 autocast、勾配チェックポイントを使い、EMAは保持しません。公式の分散学習と同一の学習レシピではありません。

初期値は1024px相当の縦横比バケット、5,000枚、実バッチ14×勾配累積2、実効バッチ28、学習率1e-5、30更新warmup、300 optimizer更新です。解像度別のflow shiftは512pxで3、1024pxで4です。

実バッチ14はA100 80GB向けです。A100 40GBのメモリ適合性は未確認で、試す場合はバッチ1から調整します。保存時のCPUメモリが必要なため、Colabの高RAMを使います。

学習データのrepository、40桁commit SHA、列名を設定してください。品質列を使わない場合は`QUALITY_COLUMN`を空欄、`MIN_AESTHETIC_TIER=0`にします。データ設定は[README](README.md#学習データを指定する)にあります。

最初の学習セルは1 optimizer更新を行い、全重み・optimizer等をDriveに保存します。この更新は本学習の1更新目です。メモリ、loss、勾配、保存結果を確認した後、学習セルを進めます。

再開用チェックポイントは1世代約18GB、BF16モデルは約6GBです。既存2世代と新しい保存が一時的に共存し、データやDriveのごみ箱も容量を使います。初期学習では約200GBの空きを推奨します。無料Driveの15GBには全体FTの再開用保存が収まりません。

途中でColabを切断した場合は、同じ`RUN_NAME`と設定で上から再実行します。保存のSHA-256を照合し、正常な世代から学習状態を復元します。目標更新数を増やしても、schedulerの過去状態を保ちます。

書き出し結果は`export/step-XXXXXXXX/model.safetensors`と`config.yaml`です。同じフォルダーに置き、公式Irisの`--checkpoint`へフォルダーを指定します。学習を再開する場合はBF16出力ではなく、`checkpoints`の全重み・optimizer入り保存を使います。

最後にDriveへの送信完了セルを実行し、GPUランタイムを接続解除します。公開版の設定・データ変換を含むCPUテストはGitHub Actionsで実行します。公開版をA100で通した実行は未検証です。
