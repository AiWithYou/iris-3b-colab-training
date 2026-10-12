# 新しいデータで全体FTを継続する

[Colabで開く](https://colab.research.google.com/github/AiWithYou/iris-3b-colab-training/blob/main/Iris3B_FullFT_Continuation.ipynb)

この公開版で作成した全体FTのチェックポイントから、新しい画像群へ学習を進めます。親runを停止し、親の保存ファイルは保持してください。途中再開でも最初に使った親チェックポイントのIDが必要です。

1. `PARENT_RUN_NAME`を指定し、`RUN_NAME`には別名を付けます。
2. 新しいデータセット、固定commit SHA、列名、枚数、品質条件を入力します。
3. seed、解像度、縦横比処理、バッチ、勾配累積、warmup、text dropout等は親と同じ設定にします。
4. `MAX_STEPS`に通算目標を指定します。例えば親が300更新なら600は追加300更新です。
5. 学習率を変える場合は`PARENT_LEARNING_RATE`に親の設定値、`LEARNING_RATE`に新しい値を入力します。optimizerのmomentとschedulerの経過は保持します。

親のFP32重み、optimizer、scheduler、乱数状態と通算の画像位置を復元し、新しいデータのローカル位置を0から始めます。新しいrepository、revision、列名、枚数、品質条件は変更できます。解像度と前処理等が一致しなければ停止します。

以前使った画像を除外する場合は、`EXCLUSION_MANIFESTS_JSON`に、例えば`["datasets/<dataset-key>/manifest.json"]`と入力します。実際の`<dataset-key>`は親データの保存フォルダーで確認します。複数フェーズ分を指定できます。

以前のmanifestのfingerprintを検証してから、画像IDと元画像bytesのSHA-256で照合します。異なるデータセット間でも画像IDが一意になるよう用意してください。除外リストと新しいデータはrunの保存・再開に結び付けて検証します。

任意の更新数で一度だけ作例を出す処理は`iris_preview_colab.preview_once(..., target_step=...)`で提供します。利用時は、指定stepの保存後に呼び出します。モデル、optimizer、scheduler、乱数、model modeを復元して学習を続けます。ノートブックは追加の途中作例を自動生成しません。

学習後は公式形式のBF16モデルを書き出し、Driveを同期してGPUを切断します。公開版のA100通し実行は未検証です。
