# fibonacci_recoverability_kl

有限1次元Fibonacci-QECCの古典的一意復元性とKnill–Laflamme（KL）訂正可能性を、同じ消失集合で比較する実行済みNotebookです。

## 開くファイル

`fibonacci_recoverability_kl.ipynb` をJupyterLab / Jupyter Notebookで開いてください。ZIPを展開し、Notebook・Pythonモジュール・`reference/`を同じフォルダ構成のまま置きます。Notebook単体では再実行できません。

実行済みの図・表を閲覧できます。再計算する場合は、必要に応じて作業用Python環境を作り、次を実行してください。

```bash
python -m pip install -r requirements.txt
```

既存JupyterでそのPython環境を選び、カーネルを再起動して上から全セルを実行してください。JupyterLab自体がない場合は `python -m pip install jupyterlab` で追加できます。単体テストは次で実行できます。

```bash
python test_recoverability_kl.py
```

## 今回確認した結果

既存候補 `000111 / 010101`、inflation 3回、N=24、軌道サイズ24と8を使用しました。

- 既存の任意消失544代表配置（m=1〜4）と連続消失16条件（m=0〜15）を再現。判定一致、旧残差は1e-12以内、旧訂正率集計との差は6.3e-17以下。
- m=0〜24の240,910試行を評価。m=0〜5と19〜24は全探索、m=6〜18は各10,000一様独立標本、random seed=42。
- **Type Bを検出。** m=2では全276配置でrecoverability成功、KL成功144配置、Type Bは132配置（47.8261%）。
- E={0,7}ではself collisionもcross collisionも0ですが、消失部分00の確率が論理0で1/12、論理1で0となり、KL diagonal条件が失敗します。
- Type Cは今回未検出。Fibonacci族でrecoverabilityが必要条件かどうかは、この計算だけでは結論できません。十分条件ではないことは具体的反例で確認しました。

詳しい説明は `FINDINGS.md` を参照してください。

## 設定

Notebook第3節で変更します。

- `PAIR_ID`: `reference/seed_pairs.jsonl` の既存候補を選択。
- `USE_MANUAL_SEEDS=True`: `SEED_0`, `SEED_1` を直接指定。同じk0/k1・巡回非同値・隣接交換条件は必ず検証。
- `LOGICAL_PAIR_FILE`: 使用する候補一覧。ファイルがない場合は `SEED_LENGTH, K0, K1` で候補を生成。基準の `reference/` 一式は既存結果の再現用に保持する。
- `INFLATION_N`: 同じ回数のinflationを両seedに適用。
- `ERASURE_SIZES=None`: m=0〜N。例えば `list(range(9))` でm=0〜8だけを解析。
- `ENUMERATION_MODE`: `auto` / `exact` / `random`。
- `MAX_EXACT_PATTERNS=100000`, `NUM_RANDOM_SAMPLES=10000`, `RANDOM_SEED=42`。

自動探索は一つのpairずつです。既存全候補を勝手に性能順位付けする処理はありません。別pairを選んだ場合も、共通基準として同梱の過去pairを再現確認します。その別pair自身の過去結果を再現したという意味ではありません。

## ファイル

- `recoverability_kl.py`: 整数KL判定、collision、解析、集計、検査表示。
- `legacy_functions.py`: 既存Notebookから抽出し再利用した関数本体。
- `test_recoverability_kl.py`: 17項目のテスト。N=9全512消失集合を独立なペア和と照合。再現失敗時の停止も検証。
- `reference/`: 既存研究の入力・比較用結果snapshot、出典とSHA256。
- `results/baseline_reproduction.csv`: 既存の訂正率再現結果。
- `results/baseline_pattern_checks.csv`: 560代表条件の個別検証。
- `results/recoverability_kl_detail.csv`: 各配置／試行の詳細。約41MiB。
- `results/recoverability_kl_summary.csv`: mごとの成功率・分類・95%区間。
- `results/type_B_C_examples.json`: ビット列、collision、KLの有理数不一致例。
- `results/run_metadata.json`, `results/unit_test_report.json`: 条件・検証結果。
- `results/*.png`, `results/*.svg`: 3種類のグラフ。

## 解釈・実装上の約束

消失位置は左から0始まり。周期境界の末尾と先頭も隣接します。`recoverable_both` は各軌道別の単射性で、論理ラベル未知のunion上の単射性とは異なります。

self collisionは異なる列の順序なし組を数え、cross collisionはO0×O1を数えます。KL diagonalは正規化された整数カウントを比較します。off-diagonalは正の等振幅なのでcross collision数0と同値です。`reduced_operator` の交差係数は表示用floatになり得ますが、KL判定は整数のみです。

ランダム抽出は復元抽出で、同じEも別試行として記録します。`sample_index`で識別できます。標本0件を不存在証明として扱いません。CSVのseed先頭0を保持するには `pd.read_csv(path, dtype={'seed_0': str, 'seed_1': str})` を指定してください。消失位置列はJSON配列です。

2^N状態ベクトル、巨大密度行列、Qiskitを使用しません。Python整数は64bitを超えるNにも対応しますが、軌道サイズおよび探索配置数による計算負荷は残ります。N>100ではm範囲と標本数を調整してください。

再実行は `results/` を更新します。以前の設定の結果も残したい場合はフォルダを複製して実行してください。
