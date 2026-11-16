# fibonacci_finite_size_scaling

同じseed pairをFibonacci inflationで拡大し、固定した有限window内の連続消失と任意消失のKL訂正率を比較するNotebookです。

## 開き方

ZIPをフォルダごと展開して、`fibonacci_finite_size_scaling.ipynb` を開いてください。Notebook単体ではなく、同じフォルダのPythonファイルと `reference/` も必要です。

Python 3.10以上の環境で、必要なら次を実行します。

```bash
python -m pip install -r requirements.txt
python -m jupyterlab fibonacci_finite_size_scaling.ipynb
```

1. 冒頭の「ここはユーザーが手動で入力する」セルでseed、inflation回数、W、mなどを設定します。
2. 最初は `RUN_FULL_ANALYSIS=False` のまま全セルを実行します。N一覧、translation test、benchmark、小規模validationを確認できます。
3. 見積りを確認し、必要に応じて条件を縮小します。本計算するときだけ `RUN_FULL_ANALYSIS=True` に変更し、先頭から再実行します。
4. 推定時間が `MAX_ESTIMATED_SECONDS` を超える場合は開始しません。benchmark確認後に時間上限を変更できます。

本番用Notebookは準備段階まで実行済みです。`fibonacci_finite_size_scaling_example.ipynb` は少数サンプルの計算例で、率・図・理論比較まで実行済みです。例の設定はn=4〜10、W=20,30,50、m=1,5,10、exact上限50、sampling各100試行です。10,000試行の本番結果とは区別してください。実行例は `RUN_FULL_ANALYSIS=True` なので、再実行すればこの限定設定で計算します。

## 継承した定義

`recoverability_kl.py` と `legacy_functions.py` は既存の `fibonacci_recoverability_kl` 成果物をそのまま同梱しています。SHA256と出典は `source_manifest.json`、元のseed variation資料との対応は `reference/provenance.json` に記録しています。

- inflationは `1 → 10, 0 → 1`。Nは完成したbit列の長さから取得します。
- 各論理状態は、重複を除いた巡回軌道の正規化等重ね合わせです。
- 同じk0/k1、巡回非同値、ring境界を含む両方向の1回隣接交換条件を検証します。
- 本計算のKL判定は既存の `LogicalPair.evaluate(E)` を呼び出します。別のKLアルゴリズムに置き換えていません。
- statevector、dense density matrixは生成しません。整数bit mask、orbit、疎カウントだけを使います。

## 統計と比較

相対パターンはW,mとrandom seedで決まる繰り返し可能な列です。Nやpairを乱数seedに含めないため、異なるサイズでも同じ試行を使います。メモリ節約のため列は必要時に再生成し、digestの一致を確認します。

samplingは独立一様な復元抽出です。重複したパターンも独立な試行として保持します。点ごとの95% Wilson区間を付けます。exactにはsampling誤差を付けません。同じ試行を共有するので異なるNの推定値は相関します。scaling表には対応のある成功・失敗の変化と、その差の標準誤差も保存します。

`R_rec_*` は自己collisionに基づく補助指標、`R_KL_*` が主指標です。summaryは全試行から逐次集計し、detailの保存件数・保存モードには左右されません。

`number_of_patterns` は両消失タイプの試行数の合計です。個別の分母は `num_patterns_arbitrary`、`num_patterns_contiguous` に記録します。

## 保存とメモリ

本番の出力先は `results/`、配布時の少数サンプルの出力先は `example_results/` です。設定セルの `OUTPUT_DIR` で変更できます。

- `finite_size_detail.csv`: 各試行の相対位置・絶対位置・recoverability・KL条件。逐次書き込み。
- `finite_size_summary.csv`: 訂正率、分母、CI、gap、failure rate、pattern digest。
- `finite_size_scaling.csv`: summaryと隣接Nの変化、対応試行の変化。
- `finite_size_windows.csv`: window全体のKL条件。`KL_correctable_window=True`なら全ての部分集合が訂正可能。
- `finite_size_thresholds.csv`: 調べたNの中での最初の全成功。exact、full-window証明、sampled観測を区別。
- `finite_size_pair_statistics.csv`: 同じn,N,W,mでのpair間mean/median/min/max/count。
- `benchmark.csv`, `small_validation.csv`, `translation_test.csv`: 準備と検証の記録。
- `theory_comparison.csv`, `convergence_assessment.csv`, `conclusions.md`: 理論比較と記述的な解釈。
- `run_metadata.json`, `notebook_config.json`: 実行条件・出典hash・sampling方式・detail保存範囲。
- `figures/`: correction rate、gap、failure、1/N、heatmap（PNG/SVG）。

`DETAIL_MODE` は `all` / `failures` / `representative` / `none` を選べます。representativeは決まった走査順の先頭から全体で指定件数を保存し、統計的に無作為な代表抽出ではありません。

画面表示用の `detail_df` は `DETAIL_MEMORY_LIMIT` 行までです。`all` と `SAVE_RESULTS=True` の組合せでは、それを超えても全detailをCSVへ保存します。保存しない場合、メモリ上限以降のdetailは残りませんが集計には含まれます。大量計算ではdisk容量も考慮して保存モードを選んでください。

## 理論と結果の範囲

[Li & Boyle, arXiv:2311.13040v2, Appendix E](https://arxiv.org/html/2311.13040v2#A5.SS1) の定義を確認しました。`f_0=f_1=1` のとき保証長は `L_theory(n)=f_n+1` です。n=6〜10では14,22,35,56,90です。

入力のseed条件と原論文の保証条件を分けています。原論文の保証には自己巡回同値類へ戻る隣接交換も除外する必要があります。n=0も保証外として扱います。詳細と確認例は `THEORY.md` を参照してください。

原論文の条件を満たす固定pairなら、固定Wは十分なinflation後に保証長に入り、その全ての部分集合が訂正可能になります。本Notebookは到達前の変化、観測threshold、理論との整合性を調べます。固定Wの結果から、ring全体に広がる任意消失の弱さまで全て有限サイズ効果と断定することはできません。

`sampled_observed` の率1は未抽出パターンの成功を証明しません。`full_window_proof` はwindow全体を直接KL判定した結果なので、その全subsetの保証です。`n_star` と `N_star` は調べたサイズ列の中の最小値であり、未計算のサイズまで含む真の最小値ではありません。

## テスト

```bash
python -m unittest -v test_finite_size_scaling
```

既存関数の同一性、inflation長、ring境界、相対パターン共有、位置変換、exact/sampleの区別とCI、window包含関係、実行前の停止条件、detailのメモリ制限、SAVE_RESULTS=False、理論の添字と適用条件を検証します。小さい系の全消失部分集合は、変更していない既存KL関数と既存の疎KL残差関数で照合します。
