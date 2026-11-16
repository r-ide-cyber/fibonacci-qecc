# fibonacci_logical_information_leakage

有限Fibonacci-QECCの消失領域について、I(R:E)、D_Z、D_X、厳密KL、classical recoverabilityを比較するNotebookです。

## 開き方

ZIPをフォルダごと展開し、`fibonacci_logical_information_leakage.ipynb` を開いてください。Notebookと同じ場所のPythonファイルも必要です。

```bash
python -m pip install -r requirements.txt
python -m jupyterlab fibonacci_logical_information_leakage.ipynb
```

1. 冒頭の手動入力セルでseed、inflation回数、window、mなどを設定します。Nは自動生成します。
2. 最初は `RUN_FULL_ANALYSIS=False` でvalidationとbenchmarkを実行します。
3. 本計算するときは `RUN_FULL_ANALYSIS=True` に変更して先頭から実行します。

初期seed `000111 / 010101` は既存のfinite-size scaling解析と同じです。指定された本番設定はn=6,7,8、W=20,30,50、m=1〜6、exact上限50,000、sampled各5,000試行です。本番Notebookは準備段階まで実行済みです。

`fibonacci_logical_information_leakage_example.ipynb` は全セル実行済みの小規模例です。n=0,1,2,4,6、W=6,20、m=1〜6、exact上限100、sampled各100試行。本番の結果とは区別してください。n=0は原論文の訂正保証の適用外です。W>Nの設定はskipped表に明記します。

## 引き継いだ定義

`legacy_functions.py`、`recoverability_kl.py`、`theory.py` は既存成果物を変更せず同梱しています。出典とSHA256は `source_manifest.json` にあります。

- inflation: `1→10, 0→1`。Nは完成したbit列の長さ。
- logical state: 重複を除いた巡回orbitの正規化等重ね合わせ。M0とM1は別々に扱います。
- 元の隣接交換条件: 等しい組成、巡回非同値、1回の01/10交換で相手の巡回同値類へ移れないこと。ring境界も含みます。
- 原論文の追加条件: 交換で自分の巡回同値類にも戻らないこと。元の条件を別名に変更せず、追加検証として実施します。
- KLとrecoverabilityは元の `LogicalPair.evaluate(E)` を毎回呼び、新しい計算と照合します。recoverabilityは各orbit内の単射性であり、cross collisionを除外しません。
- `comparison_type` の旧A/Bと、今回の重要Case A/Bは対応が逆です。旧列を維持し、今回のケースを `important_case` に保存します。

原論文：[Li & Boyle, Appendix E](https://arxiv.org/html/2311.13040v2#A5)。固定windowが保証長以下になれば、その部分集合の消失も訂正可能です。full_ringの任意消失とは区別します。

## 数値計算

本解析では2^Nのstatevectorを作りません。残存patternでorbitをgroupingし、rho00/11/01と参照系の状態を構成します。

D_Zは `0.5*||rho00-rho11||_1`、D_Xは `||rho01||_1`（追加係数なし）。どちらも0〜1、I_REは0〜2 bitです。D_Z/D_Xを足してMIに分解するわけではありません。

小さい消失領域はdense計算、大きい領域はgrouping factor・連結成分・Gram行列から厳密スペクトルを計算します。`MAX_DENSE_ERASURE_QUBITS` と `MAX_MATRIX_DIMENSION` の両方で密行列を制限します。後者の初期値1024はrho_REも対象なので、通常のdense経路はm≤9に限られ、残りはgroupingへ切り替わります。圧縮後も上限を超える場合は、近似せずMemoryErrorで停止します。

負の固有値、trace、Hermitian性、範囲、KL成功時のdecouplingを検査します。丸め前のMI・固有値診断もdetailに保存します。KLは整数カウントによる厳密判定、decoupledは数値閾値による判定なので、KL失敗でも非常に小さい漏洩が数値ゼロとなる場合はあります。

## 統計・サイズ比較

arbitraryのsamplingは一様な復元抽出で、重複試行も保持します。固定W,m,random seedでは、以前のNotebookと同じ乱数列を使い、異なるNやpairでも相対消失patternを共有します。

`calculation_mode` はexact/sampled、`operator_mode` は行列計算方式です。stdはパターン間の標準偏差（ddof=0）。sampledの最大値は観測最大で、未抽出を含む最悪値の保証ではありません。worst表は最大との差がLEAKAGE_TOL以内の異なるpatternを残し、真の数値最大と各patternの値を別列にします。

Q4のpair間比較は `LOGICAL_PAIRS` に複数pairを指定すると利用できます。相関は少なくとも3組の異なるpairと値の変動がある場合だけ表示します。初期設定の1pairからpair間の相関を主張しません。

全detailをメモリに保持するため、既定で100万行を超える計画は停止します。必要なら手動セルの `MAX_DETAIL_ROWS` を変更できます。大きな実験ではsampling数や設定数を調整してください。

## 保存物

本番の保存先は `results/`、同梱サンプルは `example_results/` です。

- `logical_leakage_detail.csv`: 1消失試行1行。位置、KL、recoverability、entropy、I_RE/D_Z/D_X、collision、診断。
- `logical_leakage_summary.csv`: mean/median/std/min/max/quantiles、率、分母。
- `logical_leakage_worst_patterns.csv`: 設定ごとの最大・同率最大pattern。
- `finite_size_leakage.csv`: 固定相対patternの軌跡、隣接Nの変化。
- `logical_leakage_important_cases.csv`: Case A–D（同じ試行が複数ケースに属する場合があります）。
- `contiguous_arbitrary_comparison.csv`: 同じpair,n,N,W,mでの消失タイプ比較。
- `logical_pair_strength.csv`, `logical_pair_correlations.csv`, `collision_leakage_summary.csv`: 補助比較。
- `validation.csv`, `benchmark.csv`, `scaling_assessment.csv`, `conclusions.md`: 検証、実測コスト、観測傾向。
- `metadata.json`: 設定、version、source hash、pattern sequence digest。
- `figures/`: 主グラフ8種のPNG/SVG。

`SAVE_RESULTS=False` ならCSV等を書きません。図の保存は独立した `SAVE_FIGURES` で制御します。

## 検証

```bash
python -m unittest -v test_leakage_core test_leakage_analysis
```

小さい系の明示的statevector・partial traceと照合し、dense/low-rankの一致、空/全消失、Type 0/Z/X/ZX、M0≠M1、N=m=1000の低rank計算、数値異常検出、元のKL・乱数列・固定pattern比較を検証しています。実施記録は `artifact_validation.json` を参照してください。

配布Notebookはこの環境の通信制限のため、カーネル用の待受socketを使わずIPython上で全コードセルを順番に実行し、その実際の出力を保存しました。Jupyter用の通常の.ipynb形式です。
