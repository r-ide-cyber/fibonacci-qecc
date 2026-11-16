# fibonacci_collision_analysis

有限1次元Fibonacci-QECCのcross/self collisionを、Hamming weight・cyclic span・support multiplicityから解析するNotebookです。

## 開き方

ZIPをフォルダごと展開してください。Notebookと同じフォルダのPythonファイルが必要です。

```bash
python -m pip install -r requirements.txt
python -m jupyterlab fibonacci_collision_analysis.ipynb
```

- `fibonacci_collision_analysis.ipynb`: 本番用。既存seed `000111 / 010101`、n=4–8、m=1–10。初期状態はvalidation・計算計画・benchmarkまで実行します。手動入力セルの `RUN_FULL_ANALYSIS=True` で本計算します。
- `fibonacci_collision_analysis_example.ipynb`: 全セル実行済み。単一pairのサイズ比較、同じ組成の7pairの比較、固定windowの例を収録しています。n=0は実装検証用です。
- `example_results/`, `example_multi_pair/`, `example_window/`: 実際に計算した表・図・設定。全探索とsamplingを明示します。
- `reference/`: 既存Notebookから引き継いだ定義・小規模比較表・出典情報。

Nは入力しません。seed pairとinflation回数から自動生成します。追加設定を含め、手動設定は冒頭の1セルにまとまっています。

実行例の研究上の判定は [research_notes.md](research_notes.md) にまとめています。N=699のgeometry実測も `benchmark_N699.json` にあります。

## 引き継いだ定義

inflationは **0→1, 1→10**。論理状態は重複を除いた巡回orbitの正の等重ね合わせで、それぞれのorbit sizeで正規化します。site 0は文字列の左端、整数のbit N−1です。

seed pairは同じk0/k1、巡回非同値、1回の隣接01/10交換で相手の巡回同値類に移らない条件を確認します。環の境界の交換も含みます。原論文のself-swapに関する追加条件は別の診断です。

`recoverable_both` は各orbit内で残存patternが一意である条件です。完全な量子訂正可能性は `KL_correctable` で別途判定します。KLは整数カウントを用いて厳密に判定し、浮動小数点の許容誤差で成功扱いにはしません。

漏洩量は既存解析と同じ `D_Z=0.5||rho00-rho11||_1`, `D_X=||rho01||_1`, `I_RE=1+S(rho_E)-S(rho_RE)` です。`COMPUTE_LEAKAGE=True` で計算できます。行列次元の上限超過時は漏洩のみNAと理由を残し、collision・KLは計算します。

## 読み取り上の重要点

期待collision**組数**はcollision**発生確率**とは異なります。同じsupportを与えるpairが多ければ組数は増えますが、同じsupportの存在eventは1つです。さらに異なるsupportのeventも重なります。

cross collisionがないことはKLのoff-diagonal条件に対応します。対角条件 `rho00=rho11` は別に必要です。self collision数やその差だけでは対角条件を判定できません。

実行例では、n=1、N=9、m=2の任意消失36通り中27通りが**cross collisionなしで対角KL不一致により失敗**します。同じ長さの連続消失は訂正可能です。したがってsmall-weight / large-spanのcross supportだけで訂正率差の全てを説明できるとは結論しません。

任意消失は全探索可能ならexact、超える場合は一様な復元抽出です。重複drawを保持し、sampled訂正率にはWilson 95%区間を付けます。windowでは固定window内に消失集合を制限します。連続消失の性能比較も同じdomainを用い、全ringのN開始位置で計算した理論期待値を別列に残します。

pair間相関はn,N,m,domainを固定して計算します。3pair未満または変数が一定なら係数はNAです。少数pairで得られた係数は記述統計であり、予測性能の検証ではありません。

## 主な保存表

| 表 | 内容 |
|---|---|
| `summary.csv` | seed、n、N、orbit size、d_cross/d_self、最小span、平均・中央値、低weight/大spanの割合 |
| `spectra.csv` | cross/selfのHamming/span spectrum、unique support spectrum |
| `joint_spectrum.csv` | weightとspanの同時分布 |
| `supports.csv` | 絶対siteのhex mask・weight・span・multiplicity・代表shift pair（位置リストは設定可能） |
| `dangerous_supports.csv` | 小weight、高multiplicityのsupport候補 |
| `low_weight_large_span_supports.csv` | 設定閾値を満たす全cross support（位置リスト付き） |
| `expected.csv` | 任意/連続消失の期待collision数、発生確率の上下界 |
| `erasure_detail.csv` | 全試行の位置、collision数、KL、recoverability、任意計算の漏洩量 |
| `rates.csv` | exact/sampledの率、分母、信頼区間、平均collision数 |
| `comparison.csv` | 任意/連続の比較、A/B/C分類、crossだけの予測の過大評価分 |
| `correlations.csv` | 同じn,N,m,domain内のpair間Pearson/Spearman |
| `finite_size.csv` | domainとmを保持したサイズ系列 |
| `leakage_correlations.csv` | 同じ設定内のpattern間のcollisionと漏洩量の関連 |
| `collision_profiles.csv` | オプションの残存patternごとのmultiplicity |

`STORE_PAIR_DETAILS=True` では全cross/self pairの詳細を `orbit_pairs.csv.gz` に保存します。標準ではunique support単位でまとめ、同じ内容の大きなpositionリストを全pairに重複保存するメモリを抑えます。全設定のCSVを保存し、代表図の設定はタイトルに示します。

`SAVE_RESULTS=False` はCSV・metadataを保存しません。図の保存は独立した `SAVE_FIGURES` で制御します。マスクは長い整数なのでCSVを表計算ソフトで開く場合も文字列として扱ってください。seed列も文字列として読みます。

## 検証と計算量

```bash
python -m unittest -v test_collision_core test_collision_kl test_collision_analysis
```

小さい系の全探索で、文字列のcollision判定・mask包含・明示的partial trace・期待値・window・周期的orbit・異なるorbit size・空/全消失を検証します。結果は `artifact_validation.json` に記録します。

pair解析は概ねorbit sizeの積。supportのweight/spanには同時巡回移動の不変性を使います。各消失集合のcollision数は残存patternをgroupingして計算し、事前に計算したsupport Counterによる判定とも照合します。各消失集合でpair XORを再計算しません。

Notebookの事前benchmarkはgeometryのみの概算です。追加API `benchmark_plan(CONFIG)` はKLの1pattern実測と概算も返します。KL、漏洩、全pattern列挙、保存コストは含まれません。本番計画が `MAX_DETAIL_ROWS` を超える場合は、設定を見直せるよう計算開始前に止めます。sampledの試行数を黙って減らすことはありません。

配布NotebookはIPythonで全コードセルを順次実行して実際の出力を保存しています。通常のJupyter Notebook形式です。

理論背景: [Li & Boyle, Appendix E](https://arxiv.org/html/2311.13040v2#A5)、[Knill & Laflamme](https://doi.org/10.1103/PhysRevA.55.900)。
