# Fibonacci arbitrary-erasure code design

日本語Notebook、計算モジュール、実行済みデモ、テストをまとめた研究用パッケージです。Fibonacci規則 `1 → 10, 0 → 1` を固定し、seed pairの設計を検討します。

## 開き方

1. ZIPを展開し、Notebookと `fibonacci_design/` を同じフォルダに置きます。
2. Python 3.10以上で `python -m pip install -r requirements.txt` を実行します。
3. `jupyter lab` で `fibonacci_arbitrary_erasure_code_design.ipynb` を開きます。
4. 冒頭で既存研究の `BASELINE_SEED_0` と `BASELINE_SEED_1`、組成、消失条件を入力して順に実行します。

baselineは提供されていないため `"..."` のままです。未入力では本計算を開始しません。動作確認は `DEMO_MODE=True` に切り替えるか、実行済み `fibonacci_arbitrary_erasure_code_design_demo.ipynb` を参照してください。

## 同梱内容

- 本Notebook：28節、39セル（code 18セル）。手動設定、4段階の探索・検証、図表、結果解釈。
- デモNotebook：全codeセル実行済み。例示baseline、長さ8・4/4、27 valid pairs、選択6候補、validation n=1,2,3。
- `fibonacci_design/`：正確なKL、leakage、collision/sequence、sampling、checkpoint、ranking、図表。
- `demo_run/results/`：デモのCSV、manifest、sample集合、checkpoint。
- `demo_run/figures/`：11種類のPNG。
- `test_fibonacci_design.py`：独立参照計算と再開・CSV再利用のテスト。

## 数学上の重要事項

有限巡回構成は [Li–Boyle, Appendix E.1](https://arxiv.org/html/2311.13040v2#A5) に合わせています。単一の隣接交換後の巡回クラスが、相手または自分の採用クラスへ戻るpairを除外します。元論文の指定がないため、ユーザーが意図する論文と同じものかは別途確認してください。

この定義で長さ6・0/1各3個の全探索をすると、有効pairは `000111 / 010101` の1組です。この組がbaselineなら、同じseed空間内では別候補を発見できません。長さ8・4/4は27組、長さ10・5/5は261組です。

logical stateは相異なる巡回列の等重み・正振幅の重ね合わせです。KL判定は縮約行列の全成分を整数比較します。`R_rec` は保持bitから元の列を一意復元できる古典的指標です。量子recoverabilityとは区別します。

I(R:E)はbits、D_ZはZ基底のtrace distance、D_XはX基底のtrace distanceです。sampledの最大leakageは観測最大値であり、真の最大値の下界です。

## 再利用と再現性

`PREVIOUS_RESULT_DIRS` に過去CSVのフォルダを指定します。同じモデル・定義・条件・sample hashを持つ本実装の記録は再利用できます。定義やsampleが確認できない旧CSVは参照用としてinventoryに残します。今回、過去解析の実ファイルは提供されていないため、旧Notebook固有の列名を推測した変換は行っていません。

任意消失のsampleはN,W,mごとに全候補共通です。sampled validationはtrainingを除いた母集団から抽出し、maskの重なりを排除します。exactなら両splitで同じ全集合を評価します。checkpointは各candidate/条件ごとに原子的に保存します。

`SAVE_RESULTS=False` は書き込みを止めます。`RESUME=True` は同一条件の計算を再利用します。並列feature計算は `N_JOBS>1` と `RESUME=False` の初回計算で利用できます。既定はserialです。

seedの先頭0を保つため、CSVは `pd.read_csv(path, dtype={"seed_0": str, "seed_1": str})` で読みます。主要CSVには比較可能性を確認する条件列が含まれます。

## 検証

同じフォルダで次を実行できます。

```sh
python -m unittest -v test_fibonacci_design.py
```

8テストを通過しました。小Nの全消失maskに対するdense partial traceとの照合、KLとI(R:E)=0の対応、collision期待値の全列挙、候補数、samplingの再現性とsplit分離、window mode、途中再開、CSVからの再利用を確認しています。

同梱デモは18 codeセルを同一IPythonプロセスで順に実行し、表と図をNotebookへ保存しました。実行環境ではJupyterのsocket起動が制限されたため、独立Jupyter kernelの起動検証は行っていません。Notebookはnbformat検証を通過しています。

デモでは改善する条件のある候補が見つかっていますが、最終採用基準を通過した候補は0です。少数sampleとleakage対象の制限を含む動作例であり、既存研究baselineに対する発見や、新しいQECCの証明を示すものではありません。

## 設計判断の範囲

weighted scoreの係数は探索設定です。相関は選別済み候補内の探索的相関で、全候補での予測精度ではありません。validation_passは指定した有限N,W,mと計算精度での判定です。無限系への外挿、因果関係の証明、振幅・位相・置換規則の変更は含みません。
